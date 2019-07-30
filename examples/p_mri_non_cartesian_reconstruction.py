"""
Neuroimaging cartesian reconstruction
=====================================

Credit: L Elgueddari

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.
"""

# Package import
import pysap
from pysap.data import get_sample_data
from mri.numerics.linear import Wavelet2
from mri.numerics.fourier import NFFT
from mri.numerics.reconstruct import sparse_rec_fista
from mri.numerics.reconstruct import sparse_rec_condatvu
from mri.numerics.gradient import Gradient_pMRI
from mri.numerics.proximity import Threshold
from pysap.plugins.mri.parallel_mri.extract_sensitivity_maps import (
    extract_k_space_center_and_locations, get_Smaps)
from pysap.plugins.mri.reconstruct.utils import normalize_frequency_locations

# Third party import
import numpy as np

# Loading input data
Il = get_sample_data("2d-pmri").data.astype("complex128")
SOS = np.squeeze(np.sqrt(np.sum(np.abs(Il)**2, axis=0)))
Smaps = np.asarray([Il[channel]/SOS for channel in range(Il.shape[0])])
kspace_loc = normalize_frequency_locations(
    get_sample_data("mri-radial-samples").data)
image = pysap.Image(data=np.abs(SOS))
image.show()


#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace
fourier_op_gen = NFFT(samples=kspace_loc, shape=SOS.shape)
kspace_data = np.asarray([fourier_op_gen.op(Il[l]) for l in
                          range(Il.shape[0])])

# Generate the senitivity matrix from undersampled data
data_thresholded, samples_thresholded = extract_k_space_center_and_locations(
    data_values=kspace_data,
    samples_locations=kspace_loc,
    thr=(0.5/128*5, 0.5/128*5),
    img_shape=SOS.shape)

Smaps, SOS_Smaps = get_Smaps(
    k_space=data_thresholded,
    img_shape=SOS.shape,
    samples=samples_thresholded,
    mode='Gridding',
    min_samples=np.min(kspace_loc),
    max_samples=np.max(kspace_loc),
    method='linear')

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the FISTA reconstruction
max_iter = 50

linear_op = Wavelet2(wavelet_name="UndecimatedBiOrthogonalTransform",
                     nb_scale=4)
prox_op = Threshold(None)
fourier_op = NFFT(samples=kspace_loc, shape=SOS.shape)
gradient_op = Gradient_pMRI(data=kspace_data,
                            fourier_op=fourier_op,
                            linear_op=linear_op,
                            S=Smaps)

x_final, transform, cost, metrics = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    prox_op=prox_op,
    cost_op=None,
    mu=1e-7,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()


#############################################################################
# Condata-Vu optimization
# -----------------------
#
# We now want to refine the zero order solution using a Condata-Vu
# optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the CONDAT-VU reconstruction
max_iter = 50
gradient_op_cd = Gradient_pMRI(data=kspace_data,
                               fourier_op=fourier_op,
                               S=Smaps)
x_final, transform, cost, metrics = sparse_rec_condatvu(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    prox_dual_op=prox_op,
    cost_op=None,
    std_est=None,
    std_est_method="dual",
    std_thr=2.,
    mu=1e-7,
    tau=None,
    sigma=None,
    relaxation_factor=1.0,
    nb_of_reweights=0,
    max_nb_of_iter=max_iter,
    add_positivity=False,
    atol=1e-4,
    verbose=1)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
