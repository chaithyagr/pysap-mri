import numpy as np

GRAPPA_RECON_AVAILABLE = False
try:
    from ggrappa.grappaND import GRAPPA_Recon
    import torch
    from ggrappa.utils import get_cart_portion_sparkling, get_grappa_filled_data_and_loc
except:
    GRAPPA_RECON_AVAILABLE = False
    
    
def do_grappa_and_append_data(kspace_shots, kspace_data, traj_params, grappa_maker):
    if not GRAPPA_RECON_AVAILABLE:
        raise ValueError("GRAPPA is not available")
    gridded_center = get_grappa_filled_data_and_loc(kspace_shots, kspace_data, traj_params)
    grappa_recon, grappa_kernel = grappa_maker(
        sig=torch.tensor(gridded_center).permute(0, 2, 3, 1),
        acs=None,
        isGolfSparks=True,
    )
    rec = rec.permute(0, 3, 1, 2).numpy()
    extra_loc, extra_data = get_grappa_filled_data_and_loc(gridded_center, grappa_recon, traj_params)
    kspace_loc = np.concatenate([kspace_loc.reshape(-1, kspace_loc.shape[-1]), extra_loc], axis=0)
    kspace_data = np.concatenate([kspace_data, extra_data], axis=0)
    return kspace_loc, kspace_data