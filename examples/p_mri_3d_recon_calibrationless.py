"""
Neuroimaging cartesian reconstruction
=====================================

Credit: Chaithya G R

In this tutorial we will reconstruct a 3D-MRI image from the sparse kspace
measurments.
"""

# Package import
import pysap
from mri.reconstruct.fourier import NFFT, Stacked3D
from mri.parallel_mri_online.gradient import Gradient_pMRI_calibrationless
from mri.numerics.linear import WaveletN
from mri.parallel_mri.cost import GenericCost
from mri.numerics.proximity import Threshold
from mri.numerics.reconstruct import sparse_rec_fista
from mri.reconstruct.utils import normalize_frequency_locations

# Third party import
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import twixreader
import datetime
import pickle as pkl
import os

def get_raw_data(filename):
    # Function that reads a SIEMENS .dat file and returns a k space data
    file = twixreader.read_twix(filename)
    measure = file.read_measurement(1)
    buffer = measure.get_meas_buffer(0)
    data = np.asarray(buffer[:])
    data = np.swapaxes(data, 1, 2)
    data = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    return data.T


def get_samples(filename):
    sample_locations = normalize_frequency_locations(loadmat(filename)['samples'])
    return sample_locations

filename = []
sparkling_file = []
filename.append(['/raw/meas_MID00451_FID13115_nsCSGRE3D_N896_FOV448_Nz64_nc75_ns2049_OS2.dat'])
sparkling_file.append(['/SPARKLING_trajectories/1stack_SPARKLING_N896_FOV448_nc75_ns2049_OS1.txt'])
filename.append(['/raw/meas_MID00452_FID13116_nsCSGRE3D_N768_FOV384_Nz64_nc75_ns2049_OS2.dat'])
sparkling_file.append(['/SPARKLING_trajectories/stack_SPARKLING_N768_nc75x2049_OS1.txt'])
filename.append(['/raw/meas_MID00453_FID13117_nsCSGRE3D_N640_FOV320_Nz64_nc75_ns2049_OS2.dat'])
sparkling_file.append(['/SPARKLING_trajectories/stack_SPARKLING_N640_nc75x2049_OS1.txt'])
filename.append(['/raw/meas_MID00454_FID13118_nsCSGRE3D_N512_FOV256_Nz64_nc75_ns2049_OS2.dat'])
sparkling_file.append(['/SPARKLING_trajectories/stack_SPARKLING_N512_nc75x2049_OS1.txt'])
filename.append(['/raw/meas_MID00455_FID13119_nsCSGRE3D_N384_FOV192_Nz64_nc75_ns2049_OS2.dat'])
sparkling_file.append(['/SPARKLING_trajectories/stack_SPARKLING_N384_nc75x2049_OS1.txt'])
filename.append(['/raw/meas_MID00463_FID13127_nsCSGRE3D_N768_FOV384_Nz64_nc64_ns2049_OS2.dat'])
sparkling_file.append(['/SPARKLING_trajectories/stack_SPARKLING_N768_nc64x2049_OS1.txt'])
filename.append(['/raw/meas_MID00462_FID13126_nsCSGRE3D_N640_FOV320_Nz64_nc54_ns2049_OS2.dat'])
sparkling_file.append(['/SPARKLING_trajectories/stack_SPARKLING_N640_nc54x2049_OS1.txt'])
filename.append(['/raw/meas_MID00458_FID13122_nsCSGRE3D_N512_FOV256_Nz64_nc43_ns2049_OS2.dat'])
sparkling_file.append(['/SPARKLING_trajectories/stack_SPARKLING_N512_nc43x2049_OS1.txt'])
filename.append(['/raw/meas_MID00457_FID13121_nsCSGRE3D_N384_FOV192_Nz64_nc32_ns2049_OS2.dat'])
sparkling_file.append(['/SPARKLING_trajectories/stack_SPARKLING_N384_nc32x2049_OS1.txt'])
filename.append(['/raw/meas_MID00459_FID13123_nsCSGRE3D_N896_FOV448_Nz64_nc75_sphere_regular_ns2049_OS2.dat'])
sparkling_file.append(['/SPARKLING_trajectories/GradientFile_samples_regular.txt'])
filename.append(['/raw/meas_MID00460_FID13124_nsCSGRE3D_N896_FOV448_Nz64_nc75_sphere_angle_ns2049_OS2.dat'])
sparkling_file.append(['/SPARKLING_trajectories/GradientFile_samples_angle.txt'])
# Loading input data
N = 448
Nz = 64
thresh = 0.05
threshz = 0.5
mu = 5e-6
max_iter = 150
kspace_loc = []
kspace_data = []
num_channels = 42

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("i",
                        help="file ID")
    args = parser.parse_args()
    i = int(args.i)
    myfile = filename[i]
    file = myfile[0]
    file_name =  os.path.splitext(os.path.basename(file))[0]
    try:
        (kspace_loc, kspace_data) = pkl.load(
            open("/neurospin/optimed/Chaithya/Data/" + file_name + ".pkl",'rb'))
    except:
        print("Could not find temp file, loading data!")
        image_name = \
            '/neurospin/optimed/Chaithya/20190802_benchmark_3D_v5' \
             + file
        mask_name = \
            '/neurospin/optimed/Chaithya/20190802_benchmark_3D_v5/samples_'+str(i+1)+'.mat'
        kspace_data = get_raw_data(image_name)
        kspace_loc = get_samples(mask_name)
        pkl.dump((kspace_loc, kspace_data), open("/neurospin/optimed/Chaithya/Data/"
                                                 + file_name + ".pkl", 'wb'),
                 protocol=4)

    linear_op = WaveletN(wavelet_name="sym8",
                         nb_scale=4, dim=3,
                         padding_mode='periodization',
                         num_channels=42)
    fourier_op = Stacked3D(fft_type='NUFFT', samples=kspace_loc,
                           shape=(N, N, Nz), platform='gpu')
    fourier = "NUFFT"
    gradient_op = Gradient_pMRI_calibrationless(data=kspace_data,
                                fourier_op=fourier_op,
                                linear_op=linear_op,
                                max_iter=10,
                                check_lips=True)
    prox_op = Threshold(mu)
    cost_synthesis = GenericCost(
        gradient_op=gradient_op,
        prox_op=prox_op,
        initial_cost=1e6,
        tolerance=1e-4,
        cost_interval=1,
        test_range=4,
        verbose=5,
        plot_output=None)

    #############################################################################
    # FISTA optimization
    # ------------------
    #
    # We now want to refine the zero order solution using a FISTA optimization.
    # Here no cost function is set, and the optimization will reach the
    # maximum number of iterations. Fill free to play with this parameter.

    # Start the FISTA reconstruction
    x_final, y_final, cost, metrics = sparse_rec_fista(
        gradient_op=gradient_op,
        linear_op=linear_op,
        prox_op=prox_op,
        cost_op=cost_synthesis,
        lambda_init=0.0,
        max_nb_of_iter=max_iter,
        atol=1e-4,
        is_multichannel=True,
        verbose=1)

    currentDT = datetime.datetime.now()
    pkl.dump((x_final, cost), open("/neurospin/optimed/Chaithya/Results/MANIAC/"
            + "calibrationless_" + file_name
            + "-N_" + str(N) + "-Nz" + str(Nz) + "-D" + str(currentDT.day) + "M"
            + str(currentDT.month) + "Y" + str(currentDT.year)
            + str(currentDT.hour) + ":" + str(currentDT.minute)
            + "_calibrationless.pkl", 'wb'), protocol=4)
