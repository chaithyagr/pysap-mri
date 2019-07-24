"""
Neuroimaging cartesian reconstruction
=====================================

Credit: L Elgueddari

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
import pysap
from pysap.data import get_sample_data
from pysap.plugins.mri.reconstruct.reconstruct import FFT2
from pysap.plugins.mri.reconstruct.reconstruct import NFFT2
from pysap.plugins.mri.parallel_mri_online.linear import Pywavelet2
from pysap.plugins.mri.parallel_mri_online.gradient import Grad2D_pMRI
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations
from pysap.plugins.mri.parallel_mri_online.reconstruct import sparse_rec_fista
from pysap.plugins.mri.parallel_mri_online.reconstruct import sparse_rec_condatvu
from pysap.plugins.mri.parallel_mri_online.proximity import MultiLevelNuclearNorm

# Third party import
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Loading input data
image_name = '/volatile/data/2017-05-30_32ch/'\
            '/meas_MID41_CSGRE_ref_OS1_FID14687.mat'
k_space_ref = loadmat(image_name)['ref']
k_space_ref /= np.linalg.norm(k_space_ref)
cartesian_reconstruction = False

if cartesian_reconstruction:
    Sl = np.zeros((32, 512, 512), dtype='complex128')
    for channel in range(k_space_ref.shape[-1]):
        Sl[channel] = np.fft.fftshift(np.fft.ifft2(np.reshape(
            k_space_ref[:, channel], (512, 512))))
    SOS = np.sqrt(np.sum(np.abs(Sl)**2, 0))
else:
    full_samples_loc = convert_mask_to_locations(np.ones((512, 512)))
    gen_image_op = NFFT2(samples=full_samples_loc, shape=(512,512))
    Sl = np.zeros((32, 512, 512), dtype='complex128')
    for channel in range(k_space_ref.shape[-1]):
        Sl[channel] = gen_image_op.adj_op(np.reshape(k_space_ref[:, channel], (512, 512)))
    SOS = np.sqrt(np.sum(np.abs(Sl)**2, 0))

mask = get_sample_data("mri-mask")
# mask.show()
image = pysap.Image(data=np.abs(SOS), metadata=mask.metadata)
# image.show()

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace
if cartesian_reconstruction:
    mask.data = np.fft.fftshift(mask.data)
    kspace_loc = convert_mask_to_locations(mask.data)
    kspace_data = []
    [kspace_data.append(mask.data * np.fft.fft2(Sl[channel]))
        for channel in range(Sl.shape[0])]
    kspace_data = np.asarray(kspace_data)
else:
    kspace_loc = convert_mask_to_locations(mask.data)
    fourier_op_1 = NFFT2(samples=kspace_loc, shape=image.shape)
    kspace_data = []
    for channel in range(Sl.shape[0]):
        kspace_data.append(fourier_op_1.op(Sl[channel]))
    kspace_data = np.asarray(kspace_data)
    # Coil compression step
    # U, s, V = np.linalg.svd(kspace_data.T, full_matrices=False)
    # kspace_data = np.dot(U[:,:16], np.dot(np.diag(s[:16]), V[:16, :]))


#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

max_iter = 10
# Start the FISTA reconstruction

linear_op = Pywavelet2('haar', nb_scale=1,
                        undecimated=False,
                        multichannel=True)

_ = linear_op.op(np.zeros(Sl.shape))

if cartesian_reconstruction:
    fourier_op = FFT2(samples=kspace_loc, shape=(512,512))
else:
    fourier_op = NFFT2(samples=kspace_loc, shape=(512, 512))


gradient_op_cd = Grad2D_pMRI(data=kspace_data,
                             linear_op=linear_op,
                             fourier_op=fourier_op)

mu_value = 1e-5
gamma = 0.5
weights = []
patch_shape = []
nb_band_scale = 4
for scale_nb in range(linear_op.nb_scale):
    for _ in range(nb_band_scale):
        weights.append(mu_value * gamma**(linear_op.nb_scale-scale_nb-1))
        patch_shape.append((2**(6 - scale_nb), 2**(6 - scale_nb), Sl.shape[0]))

        print("(Weights per scale, Patches shapes per scale)",
              (weights[-1], patch_shape[-1]))

print(patch_shape)

overlapping_factor = 2

prox_op = MultiLevelNuclearNorm(weights=weights,
                                patch_shape=patch_shape,
                                linear_op=linear_op,
                                overlapping_factor=overlapping_factor)

x_final, cost = sparse_rec_fista(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    prox_op=prox_op,
    mu=mu_value,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1,
    get_cost=True)

image_rec = pysap.Image(data=np.sqrt(np.sum(np.abs(x_final)**2, axis=0)))
image_rec.show()
plt.plot(cost)
plt.show()

gradient_op_cd = Grad2D_pMRI(data=kspace_data,
                             fourier_op=fourier_op)

linear_op = Pywavelet2('haar', nb_scale=1,
                        multichannel=True)

x_final, y_final, cost_func = sparse_rec_condatvu(
     gradient_op=gradient_op_cd,
     linear_op=linear_op,
     prox_dual_op=prox_op,
     std_est=None,
     tau=None,
     sigma=None,
     relaxation_factor=1.0,
     nb_of_reweights=0,
     max_nb_of_iter=max_iter,
     add_positivity=False,
     atol=1e-4,
     verbose=1,
     get_cost=True)

image_rec_y = pysap.Image(data=np.sqrt(np.sum(np.abs(linear_op.adj_op(
                y_final))**2, axis=0)))
image_rec_y.show()

image_rec = pysap.Image(data=np.sqrt(np.sum(np.abs(x_final)**2, axis=0)))
image_rec.show()


plt.plot(cost_func)
plt.show()
