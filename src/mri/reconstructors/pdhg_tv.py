import torch
import torch.nn as nn
from deepinv.models import TVDenoiser
from deepinv.loss.metric import PSNR
from tqdm import tqdm


class PDHG_TV(nn.Module):
    def __init__(
        self,
        lambda_reg,
        max_iter,
        lipschitz, # L_f: Lipschitz constant of ∇f (estimate via power iteration)
        data_fidelity,
        stopping_criterion=1e-5,
        relaxation_param = 1.0,
    ):
        super(PDHG_TV, self).__init__()
        self.lambd = lambda_reg
        self.steps = max_iter
        self.tau = 1.0 / lipschitz
        self.sigma = 0.9 / (self.tau * 12) # 12 is the conservative bound on ||∇||_2^2 for 3D forward differences
        self.rho = relaxation_param
        self.stopping_criterion = stopping_criterion
        self.df = data_fidelity
        self.psnr_metric = PSNR()

    @staticmethod
    def _proj_l2_ball_pointwise(p: torch.Tensor, radius: float) -> torch.Tensor:
        nrm = torch.linalg.norm(p, dim=-1, keepdim=True)
        scale = torch.clamp(nrm / radius, min=1.0)
        return p / scale

    def forward(self, y, physics, init, x_gt=None, compute_metrics=False):
        # reconstruction with Condat-Vu primal-dual hybrid gradient algorithm
        x = init
        p = torch.zeros_like(TVDenoiser.nabla(x))
        
        metrics = {"psnr": [], "residual": []}
        if compute_metrics and x_gt is not None:
            metrics["psnr"].append(self.psnr_metric(x, x_gt).item())

        for step in tqdm(range(self.steps)):
            x_old = x.clone()
            p_old = p.clone()

            # Dual: p^{k+1} = Π( p^k + σ ∇x^k )
            p = p + self.sigma * TVDenoiser.nabla(x)
            p = self._proj_l2_ball_pointwise(p, self.lambd)

            # Primal: x^{k+1} = x^k - τ( ∇f(x^k) + ∇^T(2p^{k+1}-p^k) )
            x = x - self.tau * (self.df.grad(x, y, physics) + TVDenoiser.nabla_adjoint(2.0 * p - p_old))

            # Optional relaxation (ρ in (0,2))
            x = x_old + self.rho * (x - x_old)
            p = p_old + self.rho * (p - p_old)

            rel_err = torch.linalg.norm(
                x_old.flatten() - x.flatten()
            ) / (torch.linalg.norm(x.flatten()) + 1e-12)
            
            if compute_metrics and x_gt is not None:
                metrics["psnr"].append(self.psnr_metric(x, x_gt).item())
                metrics["residual"].append(rel_err.item())
                
            if (rel_err < self.stopping_criterion):
                print("Converged at iteration:", step+1)
                break
            
        if compute_metrics and x_gt is not None:
            return x, metrics

        return x
