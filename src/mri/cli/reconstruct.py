from hydra_zen import store, zen

from mrinufft.io import read_siemens_rawdat
from mri.operators import FFT
from mri.cli.utils import raw_config, traj_config, setup_hydra_config, get_outdir_path, generate_complex_noise_cholesky, save_data_hydra
from mri.operators.fourier.utils import discard_frequency_outliers, convert_mask_to_locations
from mrinufft.io.utils import add_phase_to_kspace_with_shifts, remove_extra_kspace_samples
from mrinufft.extras.smaps import cartesian_espirit, coil_compression
from mri.reconstructors import SelfCalibrationReconstructor
from mri.reconstructors.ggrappa import do_grappa_and_append_data, GRAPPA_RECON_AVAILABLE
from mrinufft.operators.autodiff import image_as_real, image_as_cpx, kspace_as_real
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder
import torch
import tqdm
    
    
import json
import numpy as np
import pickle as pkl
import logging, os, glob
from functools import partial
import scipy as sp


log = logging.getLogger(__name__)



def dc_adjoint(obs_file: str|np.ndarray, traj_file: str, coil_compress: str|int, debug: int,
               obs_reader, traj_reader, fourier, grappa_recon=None, output_filename: str = "dc_adjoint.nii",
               return_data=False):
    """
    Reconstructs an image using the adjoint operator.

    Parameters
    ----------
    obs_file : str or np.ndarray
        Path to the observed kspace data file.
    traj_file : str
        Path to the trajectory file or the folder holding trajectory file.
        If folder is provided, the trajectory name is picked up and the data header 
        and the trajectory is obtained by searching recursively.
    obs_reader : callable
        A function that reads the observed data file and returns
        the raw data and data header.
    traj_reader : callable
        A function that reads the trajectory file and returns the trajectory
        data and parameters.
    fourier: Callable
        A Callable returning a Fourier Operator
    coil_compress : str|int, optional default -1
        The number of singular values to keep in the coil compression.
        If -1, coil compression is not applied 
    output_filename: str, optional default 'dc_adjoint.pkl'
        The output file name with the right extension.
        It can be:
        1) *.pkl / *.mat: Holds the reconstructed results saved in dictionary as `recon`.
        2) *.nii : NIFTI file holding the reconstructed images.
    grappa_af: Union[list[int], tuple[int, ...]], optional default 1
        The acceleration factor for the GRAPPA reconstruction.
        
    Returns
    -------
    None
        The reconstructed image is saved as 'dc_adjoint.pkl' file.
    """
    try:
        raw_data, data_header = obs_reader(obs_file)
    except:
        traj_file = "cart"
    if traj_file == "cart":    
        log.info("It is cartesian trajectory")
        raw_data, data_header = read_siemens_rawdat(obs_file, removeOS=True)
        mask = np.linalg.norm(raw_data, axis=0)>0
        kspace_loc = convert_mask_to_locations(mask)
        traj_params = {
            "img_size": raw_data.shape[1:],
            "num_shots": np.sum(mask[0]),
            "num_samples_per_shot": raw_data.shape[1],
        }
        kspace_data = np.ascontiguousarray(raw_data[:, mask])
    else:
        try:
            if obs_reader.keywords['slice_num'] is not None:
                data_header['slice_num'] = obs_reader.keywords['slice_num']
        except:
            pass
        log.info(f"Data Header: {data_header}")
        try:
            if not os.path.isdir(traj_file) and data_header["trajectory_name"] != os.path.basename(traj_file):
                log.warn("Trajectory file does not match the trajectory in the data file")
        except KeyError:
            log.warn("Trajectory name not found in data header, Skipped Validation")
        if os.path.isdir(traj_file):
            search_folder = traj_file
            found_trajs = glob.glob(os.path.join(search_folder, "**", data_header['trajectory_name']), recursive=True)
            if len(found_trajs) == 0:
                log.error(f"Trajectory {traj_file} from data_header not found in {search_folder}")
                exit(1)
            if len(found_trajs) > 1:
                log.warn("More than one file found, choosing first one")
            traj_file = found_trajs[0]
        elif not os.path.exists(traj_file):
            raise ValueError("Trajectory not found, exiting!")
        log.debug(f"Loading trajectory from {traj_file}")
        shots, traj_params = traj_reader(
            traj_file,
            dwell_time=traj_reader.keywords['raster_time'] / data_header["oversampling_factor"],
        )
        # Need to have image sizes as even to ensure no issues
        traj_params['img_size'] = np.asarray([
            size + 1 if size % 2 else size 
            for size in traj_params['img_size']
        ])
        log.info(f"Trajectory Parameters: {traj_params}")
        data_header["shifts"] = data_header['shifts'][:traj_params["dimension"]]
        normalized_shifts = (
            np.array(data_header["shifts"])
            / np.array(traj_params["FOV"])
            * np.array(traj_params["img_size"])
            / 1000
        )
        kspace_data = np.squeeze(raw_data).astype(np.complex64)
        kspace_loc = shots.reshape(-1, traj_params["dimension"]).astype(np.float32)
        kspace_data = remove_extra_kspace_samples(kspace_data, shots.shape[1])
        kspace_data = kspace_data.reshape(kspace_data.shape[0], -1)
        log.info(f"Phase shifting raw data for Normalized shifts: {normalized_shifts}")
        kspace_data = add_phase_to_kspace_with_shifts(
            kspace_data, kspace_loc.reshape(-1, traj_params["dimension"]), normalized_shifts
        )
        if grappa_recon is not None and np.prod(grappa_recon.keywords['af']):
            try:
                af_string = data_header['trajectory_name'].split('_G')[1].split('_')[0].split('x')
                if len(af_string) > 1 and 'd' in af_string[1]:
                    af_caipi = af_string[1].split('d')
                    af_string[1] = af_caipi[0]
                    if int(af_caipi[1])>0:
                        grappa_recon.keywords['delta'] = int(af_caipi[1])
                grappa_recon.keywords['af'] = tuple([int(float(af)) for af in af_string])
            except:
                grappa_recon.keywords['af'] = (1, )
                grappa_recon.keywords['delta'] = 0
        if grappa_recon is not None and np.prod(grappa_recon.keywords['af']) > 1:
            log.info("Performing GRAPPA Reconstruction: AF: %s", af_string)
            log.info("GRAPPA args: %s", grappa_recon.keywords)
            kspace_loc, kspace_data = do_grappa_and_append_data(
                kspace_loc,
                kspace_data,
                traj_params,
                grappa_recon,
                acs=data_header["acs"], # Pass ACS if read in data (external)
            )
        if kspace_loc.max() > 0.5 or kspace_loc.min() < 0.5:
            log.warn(f"K-space locations are above the unity range, discarding the outlier data")
            kspace_loc, kspace_data = discard_frequency_outliers(kspace_loc, kspace_data)
    if coil_compress != -1:
        log.info("Compressing coils")
        kspace_data, V = coil_compression(
            kspace_data,
            K=coil_compress,
            return_V=True,
        )
    if 'acs' in data_header and data_header['acs'] is not None:
        # Estimate the Smaps using ESPIRiT
        log.info("Estimating Smaps from ACS data using ESPIRiT")
        import cupy as cp
        acs_data = data_header['acs']
        if acs_data.shape[1] != traj_params['img_size'][0] and traj_file != "cart":
            log.warn("ACS size does not match the image size. Re-sampling")
            acs_data = sp.signal.resample(
                acs_data, traj_params['img_size'][0], axis=1
            )
        acs_data = cp.asarray(acs_data, dtype=cp.complex64)
        if coil_compress != -1:
            acs_data = (
                cp.asarray(V, dtype=cp.complex64) @ acs_data.reshape(data_header['acs'].shape[0], -1)
            ).reshape((-1, *data_header['acs'].shape[1:]))
            del V
        Smaps = cartesian_espirit(acs_data, tuple(traj_params['img_size']), decim=4).get()
        fourier.keywords['smaps'] = np.ascontiguousarray(Smaps)
        del Smaps
    if isinstance(fourier.keywords['smaps'], partial):
        fourier.keywords['smaps'] = partial(
            fourier.keywords['smaps'],
            kspace_data=kspace_data,
        )
    fourier_op = fourier(
            kspace_loc,
            tuple(int(i) for i in traj_params["img_size"]),
            n_coils=data_header["n_coils"] if coil_compress == -1 else coil_compress,
    )
    if debug > 0:
        intermediate = {
            'density_comp': fourier_op.impl.density,
            'traj_params': traj_params,
            'data_header': data_header,
            'kspace_loc': kspace_loc,
        }
        save_data_hydra('smaps.nii', fourier_op.impl.smaps)
        if coil_compress != -1:
            intermediate['kspace_data'] = kspace_data
        log.info("Saving Smaps and denisty_comp as intermediates")
        pkl.dump(intermediate, open(get_outdir_path('intermediate.pkl'), 'wb'))
    log.info("Getting the DC Adjoint")
    dc_adjoint = fourier_op.adj_op(kspace_data)
    if not fourier_op.impl.uses_sense:
        dc_adjoint = np.linalg.norm(dc_adjoint, axis=0)
    log.info("Saving DC Adjoint")
    data_header['traj_params'] = traj_params
    save_data_hydra(output_filename, dc_adjoint, data_header)
    if return_data:
        log.info("Returning data")
        return dc_adjoint, (fourier_op, kspace_data, traj_params, data_header)

def compute_analytical_sigma_ref_cupy(smaps, noise_cov):
    """
    Computes baseline R=1 noise standard deviation from coil sensitivity maps
    using GPU acceleration with CuPy.

    Parameters
    ----------
    smaps : ndarray or cp.ndarray
        Coil sensitivity maps of shape (n_coils, Nz, Ny, Nx) or (64, 256, 240, 176).
    noise_cov : ndarray or cp.ndarray
        Coil noise covariance matrix of shape (n_coils, n_coils) or (64, 64).

    Returns
    -------
    sigma_ref : ndarray
        Baseline noise SD map of shape (Nz, Ny, Nx).
    """
    import cupy as cp
    n_coils = noise_cov.shape[0]
    orig_shape = smaps.shape[1:]  # (256, 240, 176)
    n_voxels = int(np.prod(orig_shape))  # ~10.8 million voxels

    # Transfer data to GPU
    noise_cov_gpu = cp.asarray(noise_cov, dtype=cp.complex64)
    inv_cov_gpu = cp.linalg.inv(noise_cov_gpu)

    # Flatten spatial dimensions: (64, 10813440)
    if isinstance(smaps, cp.ndarray):
        smaps_flat = smaps.reshape(n_coils, n_voxels)
    else:
        smaps_flat = cp.asarray(smaps.reshape(n_coils, n_voxels), dtype=cp.complex64)

    # Compute Psi_inv @ S -> (64, n_voxels)
    # Using matrix multiplication (cp.matmul) avoids huge memory overheads of broadcast einsum
    psi_inv_s = cp.matmul(inv_cov_gpu, smaps_flat)

    # Compute S^H * (Psi_inv @ S) via element-wise dot product along coil dimension
    sh_psi_inv_s = cp.real(cp.sum(cp.conj(smaps_flat) * psi_inv_s, axis=0))

    # Free memory immediately
    del smaps_flat, psi_inv_s
    cp.get_default_memory_pool().free_all_blocks()

    # Avoid division by zero
    eps = 1e-10
    sh_psi_inv_s = cp.maximum(sh_psi_inv_s, eps)

    # Analytical R=1 noise SD
    sigma_ref_gpu = 1.0 / cp.sqrt(sh_psi_inv_s)
    sigma_ref = sigma_ref_gpu.reshape(orig_shape)

    # Return as NumPy array on CPU
    return cp.asnumpy(sigma_ref)



def gmap_recon(obs_file: str, traj_file: str, num_iterations: int, run_id: int, coil_compress: str|int, 
          debug: int, obs_reader, traj_reader, fourier, output_filename: str = "recon.nii", grappa_recon=None):
    """Reconstructs an MRI image using the given parameters.

    Parameters
    ----------
    obs_file : str
        Path to the file containing the observed k-space data.
    traj_file : str
        Path to the file containing the trajectory data.
    num_iterations : int
        Number of iterations for the reconstruction algorithm.
    coil_compress : str | int
        Method or factor for coil compression.
    algorithm : str
        Optimization algorithm to use for reconstruction.
    debug : int
        Debug level for printing debug information.
    obs_reader : callable
        Object for reading the observed k-space data.
    traj_reader : callable
        Object for reading the trajectory data.
    fourier : callable
        Object representing the Fourier operator.
    linear : callable
        Object representing the linear operator.
    sparsity : callable
        Object representing the sparsity operator.
    output_filename : str, optional
        Path to save the reconstructed image, by default "recon.pkl"
    remove_dc_for_recon: bool, optional
        Whether to remove the density compensation for reconstruction, by default True
        Note that it will still be used to estimate x_init
    validation_recon: np.ndarray, optional
        The validation reconstruction to compare the results with, by default None
    metrics: dict, optional
        List of metrics to evaluate the reconstruction, by default None
    """
    log.info("Running G-Factor map :: ")
    print(get_outdir_path())
    recon_adjoint, additional_data = dc_adjoint(
        obs_file,
        traj_file,
        coil_compress,
        debug,
        obs_reader,
        traj_reader,
        fourier,
        grappa_recon=grappa_recon,
        output_filename='dc_adj_' + output_filename,
        return_data=True,
    )
    fourier_op, kspace_data, _, data_header = additional_data
    noise_cov = np.cov(data_header['noise'].reshape(data_header['n_coils'], -1))
    L = np.linalg.cholesky(noise_cov)
    n_coils, n_samples = kspace_data.shape
    if run_id == 0:
        recon_final = fourier_op.impl.pinv_solver(kspace_data, max_iter=10).astype(np.complex64)
        save_data_hydra('pinv_' + output_filename[:-4] + '.pkl', recon_final, data_header)
        fully_sampled = generate_complex_noise_cholesky(np.prod(recon_final.shape), L=L).reshape(L.shape[0], *recon_final.shape)
        noise_recon = np.sum(np.conj(fourier_op.impl.smaps) * ifft(fully_sampled).astype(np.complex64), axis=0)
        save_data_hydra('noise_' + output_filename[:-4] + '.pkl', noise_recon, data_header)

    for i in range(num_iterations):
        complex_noise = generate_complex_noise_cholesky(n_samples, L=L)
        noisy_kspace = kspace_data + complex_noise
        rec_rep = fourier_op.impl.pinv_solver(noisy_kspace, max_iter=10).astype(np.complex64)
        rec_k = rec_rep.cpu().numpy() if hasattr(rec_rep, 'cpu') else rec_rep    
        save_data_hydra(str(run_id) + "_" + str(i) + "_" + output_filename[:-4] + '.pkl', rec_k, data_header)
    return 
    
    
def pnp_recon(obs_file: str, traj_file: str, weights_file: str, num_iterations: int, coil_compress: str|int, 
          debug: int, obs_reader, traj_reader, fourier, output_filename: str = "recon.nii", grappa_recon=None, pnp=None):
    """Reconstructs an MRI image using the given parameters.

    Parameters
    ----------
    obs_file : str
        Path to the file containing the observed k-space data.
    traj_file : str
        Path to the file containing the trajectory data.
    num_iterations : int
        Number of iterations for the reconstruction algorithm.
    coil_compress : str | int
        Method or factor for coil compression.
    algorithm : str
        Optimization algorithm to use for reconstruction.
    debug : int
        Debug level for printing debug information.
    obs_reader : callable
        Object for reading the observed k-space data.
    traj_reader : callable
        Object for reading the trajectory data.
    fourier : callable
        Object representing the Fourier operator.
    linear : callable
        Object representing the linear operator.
    sparsity : callable
        Object representing the sparsity operator.
    output_filename : str, optional
        Path to save the reconstructed image, by default "recon.pkl"
    remove_dc_for_recon: bool, optional
        Whether to remove the density compensation for reconstruction, by default True
        Note that it will still be used to estimate x_init
    validation_recon: np.ndarray, optional
        The validation reconstruction to compare the results with, by default None
    metrics: dict, optional
        List of metrics to evaluate the reconstruction, by default None
    """
    recon_adjoint, additional_data = dc_adjoint(
        obs_file,
        traj_file,
        coil_compress,
        debug,
        obs_reader,
        traj_reader,
        fourier,
        grappa_recon=grappa_recon,
        output_filename='dc_adj_' + output_filename,
        return_data=True,
    )
    fourier_op, kspace_data, _, data_header = additional_data
    log.info("Initializing PnP Reconstructor")
    recon = pnp(fourier_op, kspace_data, dc_adjoint=recon_adjoint, weights_file=weights_file, num_iterations=num_iterations)
    recon_final = recon.cpu().numpy()
    log.info("Saving reconstruction results")
    save_data_hydra(output_filename, recon_final, data_header)
    return recon
    
    
    
def recon(obs_file: str, traj_file: str, mu: float, num_iterations: int, coil_compress: str|int, 
          algorithm: str, debug: int, obs_reader, traj_reader, fourier, linear, sparsity,
          output_filename: str = "recon.nii", validation_recon: np.ndarray = None, metrics: dict = None, 
          grappa_recon=None, recon_type: str = "cs", **kwargs):
    """Reconstructs an MRI image using the given parameters.

    Parameters
    ----------
    obs_file : str
        Path to the file containing the observed k-space data.
    traj_file : str
        Path to the file containing the trajectory data.
    mu : float
        Regularization parameter for the sparsity constraint.
    num_iterations : int
        Number of iterations for the reconstruction algorithm.
    coil_compress : str | int
        Method or factor for coil compression.
    algorithm : str
        Optimization algorithm to use for reconstruction.
    debug : int
        Debug level for printing debug information.
    obs_reader : callable
        Object for reading the observed k-space data.
    traj_reader : callable
        Object for reading the trajectory data.
    fourier : callable
        Object representing the Fourier operator.
    linear : callable
        Object representing the linear operator.
    sparsity : callable
        Object representing the sparsity operator.
    output_filename : str, optional
        Path to save the reconstructed image, by default "recon.pkl"
    validation_recon: np.ndarray, optional
        The validation reconstruction to compare the results with, by default None
    metrics: dict, optional
        List of metrics to evaluate the reconstruction, by default None
    """
    recon_adjoint, additional_data = dc_adjoint(
        obs_file,
        traj_file,
        coil_compress,
        debug,
        obs_reader,
        traj_reader,
        fourier,
        grappa_recon=grappa_recon,
        output_filename='dc_adj_' + output_filename,
        return_data=True,
    )
    fourier_op, kspace_data, traj_params, data_header = additional_data
    pinv = fourier_op.impl.pinv_solver(kspace_data, max_iter=10).astype(np.complex64)
    save_data_hydra("pinv_" + output_filename, pinv, data_header)
    lipschitz_cst = fourier_op.impl.get_lipschitz_cst(100)
    if linear.func.__name__ == "WaveletN":
        linear_op = linear(shape=tuple(traj_params["img_size"]), dim=traj_params['dimension'])
        linear_op.op(pinv)
        sparse_op = sparsity(coeffs_shape=linear_op.coeffs_shape, weights=mu)
        log.info("Setting up reconstructor")
        fourier_op.impl.density = None
        reconstructor = SelfCalibrationReconstructor(
            fourier_op=fourier_op,
            linear_op=linear_op,
            regularizer_op=sparse_op,
            verbose=1,
            lipschitz_cst=lipschitz_cst,
        )
        log.info("Starting reconstruction")
        recon, costs, metrics_iter = reconstructor.reconstruct(
            kspace_data=kspace_data,
            optimization_alg=algorithm,
            x_init=pinv,
            num_iterations=num_iterations,
        )
        data_header['costs'] = costs
        data_header['metrics_iter'] = metrics_iter
    else:
        fourier_op.impl.squeeze_dims = False
        complex_out = False
        init = image_as_real(torch.from_numpy(pinv[None, None]).to("cuda")).to(torch.float32)
        kspace_data = kspace_as_real(torch.from_numpy(kspace_data[None, None]).to("cuda")).to(torch.float32)
        if linear.func.__name__ == "WaveletPrior":
            # Algorithm parameters
            physics = fourier_op.impl.make_deepinv_phy()
            iterator = optim_builder(
                iteration="FISTA",
                prior=linear(wvdim=len(fourier_op.shape)),
                data_fidelity=L2(),
                early_stop=True,
                max_iter=num_iterations,
                thres_conv=1e-3,
                params_algo={"stepsize": 0.9 * float(1/lipschitz_cst), "a": 3, "lambda": mu},
                verbose=True,
                show_progress_bar=True,
            )
            recon = iterator(
                kspace_data,    
                physics,
                init=(init, init),
            )
        elif linear.func.__name__ == "TVPrior":
            physics = fourier_op.impl.make_deepinv_phy(viewed_as_real=True)
            complex_out = True
            from mri.reconstructors.pdhg_tv import PDHG_TV
            solver_tv = PDHG_TV(
               lambda_reg=mu,
               max_iter=num_iterations,
               lipschitz=lipschitz_cst,
               data_fidelity=L2(),
               stopping_criterion=1e-3,
            )
            recon = solver_tv(kspace_data, physics, init=init, compute_metrics=False)
    if complex_out:
        recon = image_as_cpx(recon.cpu()).numpy().squeeze()
    save_data_hydra(output_filename, recon, data_header)
    return recon, fourier_op.impl.smaps

setup_hydra_config()
store(
    dc_adjoint,
    obs_reader=raw_config,
    traj_reader=traj_config,
    coil_compress=-1,
    debug=0,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu"},
        {"fourier/density_comp": "pipe"},
        {"grappa_recon": "enable"} if GRAPPA_RECON_AVAILABLE else {},
        {"fourier/smaps": "low_frequency"},
    ],
    name="dc_adjoint",
)
store(
    recon,
    obs_reader=raw_config,
    traj_reader=traj_config,
    algorithm="pogm",
    num_iterations=10,
    coil_compress=-1,
    mu=1e-7,
    debug=0,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu"},
        {"fourier/density_comp": "pipe"},
        {"grappa_recon": "enable"} if GRAPPA_RECON_AVAILABLE else {},
        {"fourier/smaps": "low_frequency"},
        {"linear": "deepinv_wv"},
        {"sparsity": "weighted_sparse"},
    ],
    name="recon",
)
store(
    recon,
    obs_reader=raw_config,
    traj_reader=traj_config,
    algorithm="pogm",
    num_iterations=10,
    coil_compress=10,
    mu=1e-7,
    debug=0,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu_lowmem"},
        {"grappa_recon": "enable"} if GRAPPA_RECON_AVAILABLE else {},
        {"fourier/density_comp": "pipe_lowmem"},
        {"fourier/smaps": "espirit"},
    ],
    name="recon_lowmem",
)
store(
    pnp_recon,
    obs_reader=raw_config,
    traj_reader=traj_config,
    coil_compress=-1,
    debug=0,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu"},
        {"fourier/density_comp": "pipe"},
        {"grappa_recon": "enable"} if GRAPPA_RECON_AVAILABLE else {},
        {"fourier/smaps": "espirit"},
        {"pnp": "gpu"}
    ],
    name="pnp_recon",
)
store(
    gmap_recon,
    obs_reader=raw_config,
    traj_reader=traj_config,
    coil_compress=-1,
    run_id=0,
    num_iterations=10,
    debug=0,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu"},
        {"fourier/density_comp": "pipe"},
        {"grappa_recon": "enable"} if GRAPPA_RECON_AVAILABLE else {},
        {"fourier/smaps": "espirit"},
    ],
    name="gmap_recon",
)

# Setup the Hydra Config and callbacks.
store.add_to_hydra_store()


def run_recon():
    zen(recon).hydra_main(
        config_name="recon",
        config_path=None,
        version_base="1.3",
    )

def run_gmap_recon():
    zen(gmap_recon).hydra_main(
        config_name="gmap_recon",
        config_path=None,
        version_base="1.3",
    )

def run_pnp_recon():
    zen(pnp_recon).hydra_main(
        config_name="pnp_recon",
        config_path=None,
        version_base="1.3",
    )
    
def run_adjoint():
    zen(dc_adjoint).hydra_main(
        config_name="dc_adjoint",
        config_path=None,
        version_base="1.3",
    )
