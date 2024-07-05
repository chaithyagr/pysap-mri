from hydra_zen import store, zen
import numpy as np

from mri.cli.utils import traj_config
from mri.operators.fourier.utils import discard_frequency_outliers
import nibabel as nib
import logging
from mri.cli.reconstruct import recon
from mri.optimizers.utils.metrics import box_psnr, box_ssim


log = logging.getLogger(__name__)


def retro(obs_file: str, traj_file: str, mu: float, num_iterations: int, coil_compress: str|int, 
          algorithm: str, debug: int, traj_reader, fourier, forward, linear, sparsity,
          output_filename: str = "recon.pkl"):
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
    sparsity : callable
        Sparsity operator function.
    output_filename : str, optional
        Output filename for the reconstructed data, by default "recon.pkl".
    """
    image = nib.load(obs_file).get_fdata(dtype=np.complex64)
    shots, traj_params = traj_reader(
        traj_file,
        dwell_time='min_osf',
    )
    kspace_loc = discard_frequency_outliers(shots.reshape(-1, traj_params["dimension"]))
    forward_op = forward(kspace_loc, traj_params["img_size"], n_coils=image.shape[0])
    kspace_data = forward_op.op(image)
    data_header = {
        "n_coils": image.shape[0],
        "shifts": [0, 0, 0],
        "type": "retro_recon",
        "n_adc_samples": shots.shape[1]*traj_params['min_osf'],
        "n_slices": 1,
        "n_contrasts": 1,
    }
    recon(
        obs_file="",
        traj_file=traj_file,
        mu=mu,
        num_iterations=num_iterations,
        coil_compress=coil_compress,
        algorithm=algorithm,
        debug=debug,
        obs_reader=lambda x: (kspace_data, data_header),
        traj_reader=traj_reader,
        fourier=fourier,
        linear=linear,
        sparsity=sparsity,
        output_filename=output_filename,
        validation_recon=np.linalg.norm(image, axis=0),
        metrics={
            "psnr": box_psnr,
            "ssim": box_ssim,
        }
    )

store(
    retro,
    traj_reader=traj_config,
    algorithm="pogm",
    num_iterations=30,
    coil_compress=5,
    debug=1,
    hydra_defaults=[
        "_self_",
        {"forward": "gpu"},
        {"fourier": "gpu"},
        {"fourier/density_comp": "pipe_lowmem"},
        {"fourier/smaps": "low_frequency"},
        
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

