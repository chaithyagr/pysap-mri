from .drunet import load_drunet_mri
from deepinv.physics import LinearPhysics
import numpy as np
from deepinv.optim.data_fidelity import L2
from deepinv.optim import optim_builder, PnP


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
    


def pnp_reconstruct(fourier_op, kspace_data, traj_params, data_header):
    physics  = Nufft(fourier_op)
    denoiser = load_drunet_mri("/volatile/")
    prior = PnP(denoiser)
    max_iter = 50
    kwargs_optim = dict()
    kwargs_optim["params_algo"] = get_DPIR_params(
            s1=0.1,
            s2=0.05,
            lamb=2,
            n_iter=max_iter,
        )
    algo = optim_builder(
            iteration="HQS",
            prior=prior,
            data_fidelity=L2(),
            early_stop=False,
            custom_init=get_custom_init,
            max_iter=max_iter,
            verbose=False,
            **kwargs_optim,
        )
    for itr in range(max_iter):
        x_cur = algo.fixed_point.single_iteration(
                x_cur,
                itr,
                kspace_data,
                physics,
                compute_metrics=False,
                x_gt=None,
            )
    return x_cur
        

    
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
