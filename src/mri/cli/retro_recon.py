from hydra_zen import store, zen
import numpy as np

from mri.cli.utils import traj_retro_config, fourier_op_config, generate_complex_noise_cholesky, save_data_hydra
from mri.operators.fourier.utils import discard_frequency_outliers
import nibabel as nib
import logging, os
from mri.cli.reconstruct import recon
from mrinufft.io import read_siemens_rawdat
from mrinufft.extras.cartesian import ifft, fft
from mri.optimizers.utils.metrics import box_psnr, box_ssim

log = logging.getLogger(__name__)


def retro(obs_file: str, traj_file: str, mu: float, num_iterations: int, coil_compress: str|int, 
          algorithm: str, debug: int, traj_reader, fourier, forward, linear, 
          noise_cov: str = None, output_filename: str = "recon.nii"):
    """Perform retrospective reconstruction on MRI data.
    This function takes MRI data and performs retrospective reconstruction using the specified parameters.

    Parameters
    ----------
    obs_file : str
        Path to the observed MRI data file.
    traj_file : str
        Path to the trajectory file.
    mu : float
        Regularization parameter.
    num_iterations : int
        Number of iterations for the reconstruction algorithm.
    coil_compress : str | int
        Coil compression method or factor.
    algorithm : str
        Reconstruction algorithm to use.
    debug : int
        Debug level for the reconstruction process.
    traj_reader : callable
        Function to read the trajectory file.
    fourier : callable
        Fourier transform function.
    forward : callable
        Forward operator function.
    linear : callable
        Linear operator function.
    output_filename : str, optional
        Output filename for the reconstructed data, by default "recon.pkl".
    """
    shots, traj_params = traj_reader(traj_file)
    dwell_time_acquired = 0.01/traj_params['min_osf'] if traj_reader.keywords['dwell_time'] == 'min_osf' else traj_reader.keywords['dwell_time']
    obs_time = dwell_time_acquired * shots.shape[1]
    shots = np.clip(shots, -0.5, 0.5)
    if obs_file.lower().endswith((".dat")):
        cart_data, header = read_siemens_rawdat(obs_file, removeOS=True)
        image = ifft(cart_data).astype(np.complex64)
        affine = header['affine']
        noise_cov = np.cov(header['noise'].reshape(header['n_coils'], -1))
        NOISE_REF_DWELL_TIME_MS = 5e-3
        # Calculate time spent per Nyquist voxel
        time_per_nyquist_voxel_cartesian =  obs_time / traj_params['img_size'][0] 
        noise_factor = NOISE_REF_DWELL_TIME_MS * ( 1/ dwell_time_acquired - 1 / time_per_nyquist_voxel_cartesian)
        noise_cov *= noise_factor
    else:
        nifty = nib.load(obs_file)
        affine = nifty.affine
        image = nifty.get_fdata(dtype=np.complex64)
        cart_data = fft(image)
        if noise_cov is not None:
            log.info("Adding noise to the k-space data")
            noise_cov = np.load(noise_cov)
    mid_point = np.asarray(cart_data.shape[-2:]) // 2
    acs = cart_data[:, :, mid_point[0]-12:mid_point[0]+12, mid_point[1]-12:mid_point[1]+12]
    kspace_loc = shots.reshape(-1, traj_params["dimension"]).astype(np.float32)
    forward_op = forward(kspace_loc, traj_params["img_size"], n_coils=image.shape[0])
    # Dont normalize, to ensure energy is preserved. This is important for noise addition and SNR calculations.
    kspace_data = forward_op.op(image) * np.sqrt(2**len(traj_params["img_size"])) 
    noise = generate_complex_noise_cholesky(kspace_data.shape[1], noise_cov=noise_cov)
    kspace_data += noise

    data_header = {
        "n_coils": image.shape[0],
        "shifts": [0, 0, 0],
        "type": "retro_recon",
        "n_adc_samples": shots.shape[1],
        "n_slices": 1,
        "n_contrasts": 1,
        "oversampling_factor": int(np.around(0.01 / dwell_time_acquired)),
        "trajectory_name": os.path.basename(traj_file),
        "acs": acs,
        "affine": affine,
    }
    recon_image, smaps = recon(
        obs_file="",
        traj_file=traj_file,
        mu=mu,
        num_iterations=num_iterations,
        coil_compress=coil_compress,
        algorithm=algorithm,
        debug=debug,
        obs_reader=lambda x: (kspace_data.reshape(kspace_data.shape[0], shots.shape[0], -1), data_header),
        traj_reader=traj_reader,
        fourier=fourier,
        linear=linear,
        sparsity=None,
        output_filename=output_filename,
        validation_recon=np.linalg.norm(image, axis=0),
        metrics={
            "psnr": box_psnr,
            "ssim": box_ssim,
        }
    )
    gt = np.sum(np.conj(smaps) * image, axis=0)
    save_data_hydra("gt_" + output_filename, gt, data_header)
    box_psnr_val = box_psnr(recon_image, gt)
    box_ssim_val = box_ssim(recon_image, gt)
    log.info(f"Box PSNR: {box_psnr_val:.2f} dB, Box SSIM: {box_ssim_val:.4f}")

store(
    retro,
    traj_reader=traj_retro_config,
    algorithm="pogm",
    num_iterations=30,
    forward=fourier_op_config,
    coil_compress=-1,
    debug=1,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu"},
        {"fourier/density_comp": "pipe_lowmem"},
        {"fourier/smaps": "low_frequency"},
        {"linear": "deepinv_tv"},
    ],
    name="retro_recon",
)

# Setup the Hydra Config and callbacks.
store.add_to_hydra_store()


def run_retro_recon():
    zen(retro).hydra_main(
        config_name="retro_recon",
        config_path=None,
        version_base="1.3",
    )
