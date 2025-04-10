from .drunet import load_drunet_mri
from deepinv.physics import LinearPhysics
import numpy as np
from deepinv.optim.data_fidelity import L2
from deepinv.optim import optim_builder, PnP
import torch
from tqdm import tqdm

class Nufft(LinearPhysics):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def __init__(
        self,
        nufft_op,
        **kwargs
    ):
        super(Nufft, self).__init__(**kwargs)
        self.nufft = nufft_op
        
    def A(self, x):
        return self.nufft.op(x)

    def A_adjoint(self, kspace):
        return self.nufft.adj_op(kspace)
    


def pnp_reconstruct(fourier_op, kspace_data, dc_adjoint, weights_file: str, start_sigma: float = 0.5,
                    end_sigma: float = 0.01, lamda: float = 2, max_iter: int = 10, device: str = "cpu"):
    physics  = Nufft(fourier_op)
    dc_adjoint = torch.from_numpy(dc_adjoint).to(device)
    denoiser = load_drunet_mri(weights_file, norm_factor=1/float(dc_adjoint.abs().max().cpu()), device=device)
    prior = PnP(denoiser)
    kwargs_optim = dict()
    kwargs_optim["params_algo"] = get_DPIR_params(
        s1=start_sigma,
        s2=end_sigma,
        lamb=lamda,
        n_iter=max_iter,
    )
    algo = optim_builder(
        iteration="HQS",
        prior=prior,
        data_fidelity=L2(),
        early_stop=False,
        custom_init=lambda y, phy: {"est": (dc_adjoint, dc_adjoint.detach().clone())},
        max_iter=max_iter,
        verbose=False,
        **kwargs_optim,
    )
    algo.fixed_point.show_progress_bar = True
    kspace_data = torch.from_numpy(kspace_data).to(device)
    x_est = algo(kspace_data, physics=physics)
    return x_est
        

    
def get_custom_init(y, physics):
    est = physics.A_dagger(y)
    return {"est": (est, est.detach().clone())}


def get_DPIR_params(s1=0.5, s2=0.01, lamb=2, n_iter=10):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), n_iter).astype(np.float32)
    stepsize = (sigma_denoiser / max(0.001, s2)) ** 2
    return {"lambda": lamb, "g_param": list(sigma_denoiser), "stepsize": list(stepsize)}
