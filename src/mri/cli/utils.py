from hydra_zen import store, builds
from hydra.conf import HydraConf, JobConf, SweepDir
import hydra

from mrinufft.io import read_trajectory
from mri.operators import NonCartesianFFT, WaveletN
from mri.optimizers.utils.cost import GenericCost
from mri.operators.fourier.utils import estimate_density_compensation
from mrinufft.io.nsp import read_arbgrad_rawdat, read_siemens_rawdat
from mrinufft.extras.smaps import get_smaps
from mri.operators import NonCartesianFFT, WeightedSparseThreshold
from modopt.opt.linear import Identity
import os
from mri.io.output import save_data
from deepinv.optim.prior import WaveletPrior, TVPrior
save_data_hydra = lambda x, *args, **kwargs: save_data(get_outdir_path(x), *args, **kwargs)


try:
    from ggrappa.grappaND import GRAPPA_Recon
    grappa_config = builds(
        GRAPPA_Recon,
        zen_exclude=["sig", "acs", "isGolfSparks", "quiet"],
        zen_partial=True,        
        populate_full_signature=True,
    )
    grappa_store = store(group="grappa_recon")
    grappa_store(grappa_config, name="enable", af=1)
    grappa_store(grappa_config, name="disable", af=0)
    
except:
    pass


from mri.reconstructors.pnp import pnp_reconstruct
pnp_config = builds(
    pnp_reconstruct,
    zen_exclude=["fourier_op", "kspace_data", "dc_adjoint", "weights_file", "num_iterations"],
    zen_partial=True,
    populate_full_signature=True,
)
pnp_store = store(group="pnp")
pnp_store(pnp_config, name="cpu")
pnp_store(pnp_config, name="gpu", device="cuda")
        
    
raw_config = builds(read_arbgrad_rawdat, populate_full_signature=True, zen_partial=True)
cart_config = builds(read_siemens_rawdat, populate_full_signature=True, zen_partial=True)

traj_config = builds(
    read_trajectory,
    populate_full_signature=True,
    zen_exclude=["dwell_time"],
    zen_partial=True,
)
traj_retro_config = builds(
    read_trajectory,
    populate_full_signature=True,
    zen_partial=True,
)


density_est_config = builds(
    estimate_density_compensation,
    populate_full_signature=True,
    zen_exclude=['kspace_loc', 'volume_shape'],
    zen_partial=True,
)
smaps_config = builds(
    get_smaps("low_frequency"),
    populate_full_signature=True,
    blurr_factor=30.0,
    # We estimate density, with separate args. It is passed by compute_smaps in mri-nufft
    zen_exclude=["density"],
    zen_partial=True,
)
smaps_espirit_config = builds(
    get_smaps("espirit"),
    populate_full_signature=True,
    zen_partial=True,
)
fourier_op_config = builds(
    NonCartesianFFT,
    populate_full_signature=True,
    implementation="gpuNUFFT",
    zen_exclude=["n_coils"],
    zen_partial=True,
)

# Regularizers
wavelet_local_config = builds(
    WaveletN,
    populate_full_signature=True,
    zen_partial=True,
    wavelet_name="sym8",
    nb_scale=3,
    zen_exclude=["shape"]
)
wavelet_deepinv_config = builds(
    WaveletPrior,
    populate_full_signature=True,
    zen_partial=True,
    wv="sym8",
    level=3,
    is_complex=True,
    device="cuda",
    zen_exclude=["wvdim"],
)
tv_deepinv_config = builds(
    TVPrior,
    populate_full_signature=True,
    zen_partial=True,
    n_it_max=5,
)
sparsity_config = builds(
    WeightedSparseThreshold,
    populate_full_signature=True,
    zen_partial=True,
    linear=Identity(),
    use_gpu=True,
    zen_exclude=["coeffs_shape", "linear", "weights", "use_gpu"]
)
cost_config = builds(
    GenericCost,
    cost_interval=None,
    test_range=4,
    verbose=0,
)

fourier_store = store(group="fourier")
fourier_store(fourier_op_config, name="gpu")
fourier_store(
    fourier_op_config,
    implementation="finufft",
    name="cpu",
)
fourier_store(
    fourier_op_config,
    upsampfac=1,
    implementation="gpuNUFFT",
    name="gpu_lowmem",
)
smaps_store = store(group="fourier/smaps")
smaps_store(smaps_config, name="low_frequency")
smaps_store(smaps_espirit_config, name="espirit")
density_store = store(group="fourier/density_comp")
density_store(density_est_config, implementation="pipe", name="pipe")
density_store(density_est_config, implementation="pipe", osf=1, name="pipe_lowmem")

linear_store = store(group="linear")
linear_store(wavelet_local_config, name="cpu_wv")
linear_store(wavelet_deepinv_config, name="deepinv_wv")
linear_store(tv_deepinv_config, name="deepinv_tv")

sparsity_store = store(group="sparsity")
sparsity_store(sparsity_config, name="weighted_sparse")


def setup_hydra_config(verbose=False, multirun_gather=False):
    """
    Set up the configuration for Hydra.

    Parameters
    ----------
    verbose : bool, optional
        If True, the verbose mode is enabled, by default False
    multirun_gather : bool, optional
        If True, the multirun gather is enabled, by default False

    Returns
    -------
    None
        This function does not return anything.
    """
    outdir = os.environ.get('RECON_OUTDIR', 'recon')
    callbacks = {
        'git_infos': {
            '_target_': "hydra_callbacks.GitInfo",
            'clean': True
        },
        'resource_monitor': {
            '_target_': "hydra_callbacks.ResourceMonitor",
            'sample_interval': 1,
            'gpu_monit': True,
        },
        'runtime_perf': {
            '_target_': "hydra_callbacks.RuntimePerformance"
        },
    }
    if multirun_gather:
        callbacks['multirun_gather'] = {
                '_target_': "hydra_callbacks.MultiRunGatherer",
            'result_file': "metrics.json",
        }
    store(
        HydraConf(
            job=JobConf(name="recon"),
            sweep=SweepDir(dir=os.path.join(outdir, "${hydra.job.name}") + "/${now:%Y-%m-%d-%H-%M-%S}", subdir="${obs_file}"),
            callbacks=callbacks,
            verbose=verbose,
        )
    )

def get_outdir_path(filename=''):
    """Get the output directory path.

    This function returns the path of the output directory where the files will be saved.

    Parameters
    ----------
    filename : str, optional
        The name of the file to be appended to the output directory path, by default ''

    Returns
    -------
    str
        The path of the output directory.

    """
    out = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if filename != '':
        out = os.path.join(out, filename)
    return out

def generate_complex_noise_cholesky(n_samples, noise_cov=None, L=None):
    import numpy as np
    if L is None:
        L = np.linalg.cholesky(noise_cov)
    n_coils = L.shape[0]
    # 2. Standard complex Gaussian noise ~ CN(0, I)
    z = (np.random.randn(n_coils, n_samples) + 1j * np.random.randn(n_coils, n_samples)) / np.sqrt(2)
    # 3. Correlate channels
    return L @ z