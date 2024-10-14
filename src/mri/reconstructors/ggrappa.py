import numpy as np

GRAPPA_RECON_AVAILABLE = False
try:
    from ggrappa.grappaND import GRAPPA_Recon
    import torch
    from ggrappa.utils import get_cart_portion_sparkling, get_grappa_filled_data_and_loc
    GRAPPA_RECON_AVAILABLE = True
except:
    pass
    
    
def do_grappa_and_append_data(kspace_loc, kspace_data, traj_params, grappa_maker):
    kspace_shots = kspace_loc.reshape(traj_params['num_shots'], -1, traj_params['dimension'])
    if not GRAPPA_RECON_AVAILABLE:
        raise ValueError("GRAPPA is not available")
    gridded_center = get_cart_portion_sparkling(kspace_shots, traj_params, kspace_data)
    grappa_recon, grappa_kernel = grappa_maker(
        sig=torch.tensor(gridded_center).permute(0, 2, 3, 1),
        acs=None,
        isGolfSparks=True,
    )
    grappa_recon = grappa_recon.permute(0, 3, 1, 2).numpy()
    extra_loc, extra_data = get_grappa_filled_data_and_loc(gridded_center, grappa_recon, traj_params)
    kspace_loc = np.concatenate([kspace_loc, extra_loc], axis=0)
    kspace_data = np.hstack([kspace_data, extra_data])
    return kspace_loc, kspace_data