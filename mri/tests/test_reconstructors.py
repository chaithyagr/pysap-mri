# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Third-party import
import numpy as np
import unittest

# Package import
from mri.generators import KspaceGeneratorBase
from mri.operators.fourier.cartesian import FFT
from mri.operators.fourier.non_cartesian import NonCartesianFFT, Stacked3DNFFT
from mri.operators.proximity.weighted import WeightedSparseThreshold
from mri.operators.linear.wavelet import WaveletUD2, WaveletN
from mri.operators.proximity.ordered_weighted_l1_norm import OWL
from mri.reconstructors import SingleChannelReconstructor, \
    SelfCalibrationReconstructor, CalibrationlessReconstructor
from mri.operators.utils import convert_mask_to_locations
from pysap.data import get_sample_data

from itertools import product
from modopt.opt.proximity import GroupLASSO
from modopt.opt.linear import Identity


class TestReconstructor(unittest.TestCase):
    """ Tests every reconstructor with mu=0, a value to which we know the
    solution must converge to analytical solution,
    ie the inverse fourier transform
    """
    def setUp(self):
        """ Setup common variables to be used in tests:
        num_iter : Number of iterations
        images : Ground truth images to test with, obtained from server
        mask : MRI fourier space mask
        decimated_wavelets : Decimated wavelets to test with
        undecimated_wavelets : Undecimated wavelets to test with
        optimizers : Different optimizers to test with
        nb_scales : Number of scales
        test_cases : holds the final test cases
        """
        self.num_iter = 40
        # TODO getting images from net slows down these tests,
        #  we would prefer to rather use random complex data.
        self.images = [get_sample_data(dataset_name="mri-slice-nifti")]
        print("[info] Image loaded for test: {0}.".format(
            [im.data.shape for im in self.images]))
        self.mask = get_sample_data("mri-mask").data
        # From WaveletN
        self.decimated_wavelets = ['sym8']
        # From WaveletUD2, tested only for analysis formulation
        self.undecimated_wavelets = [24]
        self.recon_type = ['cartesian', 'non-cartesian']
        self.optimizers = ['fista', 'condatvu', 'pogm']
        self.nb_scales = [4]
        self.test_cases = list(product(
                self.images,
                self.nb_scales,
                self.optimizers,
                self.recon_type,
                self.decimated_wavelets,
            ))
        self.test_cases += list(product(
                self.images,
                self.nb_scales,
                ['condatvu'],
                self.recon_type,
                self.undecimated_wavelets,
            ))

    @staticmethod
    def get_linear_n_regularization_operator(
            wavelet_name,
            image_shape, dimension=2, nb_scale=3,
            n_coils=1, n_jobs=1, verbose=0):
        # A helper function to obtain linear and regularization operator
        try:
            linear_op = WaveletN(
                nb_scale=nb_scale,
                wavelet_name=wavelet_name,
                dim=dimension,
                n_coils=n_coils,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        except ValueError:
            # TODO this is a hack and we need to have a separate WaveletUD2.
            # For Undecimated wavelets, the wavelet_name is wavelet_id
            linear_op = WaveletUD2(
                wavelet_id=wavelet_name,
                nb_scale=nb_scale,
                n_coils=n_coils,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        linear_op.op(np.squeeze(np.zeros((n_coils, *image_shape))))
        regularizer_op = WeightedSparseThreshold(
            linear=Identity(),
            weights=0,
            coeffs_shape=linear_op.coeffs_shape,
            thresh_type="soft"
        )
        return linear_op, regularizer_op

    def test_single_channel_reconstruction(self):
        """ Test all the registered transformations for
        single channel reconstructor.
        """
        print("Process test for SingleChannelReconstructor ::")
        for i in range(len(self.test_cases)):
            print("Test Case " + str(i) + " " + str(self.test_cases[i]))
            image, nb_scale, optimizer, recon_type, name = self.test_cases[i]
            if optimizer == 'condatvu':
                formulation = "analysis"
            else:
                formulation = "synthesis"
            if recon_type == 'cartesian':
                fourier = FFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape)
            else:
                fourier = NonCartesianFFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape)
            kspace_data = fourier.op(image.data)
            linear_op, regularizer_op = \
                self.get_linear_n_regularization_operator(
                    wavelet_name=name,
                    dimension=len(fourier.shape),
                    nb_scale=3,
                    image_shape=image.shape,
                )
            reconstructor = SingleChannelReconstructor(
                fourier_op=fourier,
                linear_op=linear_op,
                regularizer_op=regularizer_op,
                gradient_formulation=formulation,
                verbose=0,
            )
            x_final, costs, _ = reconstructor.reconstruct(
                kspace_data=kspace_data,
                optimization_alg=optimizer,
                num_iterations=self.num_iter,
            )
            fourier_0 = FFT(
                samples=convert_mask_to_locations(self.mask),
                shape=image.shape,
            )
            data_0 = fourier_0.op(image.data)
            # mu is 0 for above single channel reconstruction and
            # hence we expect the result to be the inverse fourier transform
            np.testing.assert_allclose(
                x_final, fourier_0.adj_op(data_0), rtol=1e-3)

    def test_self_calibrating_reconstruction(self):
        """ Test all the registered transformations.
        """
        self.num_channels = 2
        print("Process test for SelfCalibratingReconstructor ::")
        for i in range(len(self.test_cases)):
            print("Test Case " + str(i) + " " + str(self.test_cases[i]))
            image, nb_scale, optimizer, recon_type, name = self.test_cases[i]
            image_multichannel = np.repeat(image.data[np.newaxis],
                                           self.num_channels, axis=0)
            if optimizer == 'condatvu':
                formulation = "analysis"
            else:
                formulation = "synthesis"
            if recon_type == 'cartesian':
                fourier = FFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape,
                    n_coils=self.num_channels,
                )
            else:
                fourier = NonCartesianFFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape,
                    n_coils=self.num_channels,
                )
            kspace_data = fourier.op(image_multichannel)
            linear_op, regularizer_op = \
                self.get_linear_n_regularization_operator(
                    wavelet_name=name,
                    dimension=len(fourier.shape),
                    nb_scale=2,
                    n_coils=1,
                    image_shape=image.shape,
                )
            reconstructor = SelfCalibrationReconstructor(
                fourier_op=fourier,
                linear_op=linear_op,
                regularizer_op=regularizer_op,
                gradient_formulation=formulation,
                verbose=0,
            )
            x_final, costs, _ = reconstructor.reconstruct(
                kspace_data=kspace_data,
                optimization_alg=optimizer,
                num_iterations=self.num_iter,
            )
            fourier_0 = FFT(
                samples=convert_mask_to_locations(self.mask),
                shape=image.shape,
                n_coils=self.num_channels,
            )
            recon = fourier_0.adj_op(fourier_0.op(image_multichannel))
            np.testing.assert_allclose(
                np.abs(x_final),
                np.sqrt(np.sum(np.abs(recon)**2, axis=0)),
                rtol=1e-3
            )

    def test_calibrationless_reconstruction(self):
        """ Test all the registered transformations.
        """
        self.num_channels = 2
        print("Process test for SparseCalibrationlessReconstructor ::")
        for i in range(len(self.test_cases)):
            print("Test Case " + str(i) + " " + str(self.test_cases[i]))
            image, nb_scale, optimizer, recon_type, name = self.test_cases[i]
            image_multichannel = np.repeat(image.data[np.newaxis],
                                           self.num_channels, axis=0)
            if optimizer == 'condatvu':
                formulation = "analysis"
            else:
                formulation = "synthesis"
            if recon_type == 'cartesian':
                fourier = FFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape,
                    n_coils=self.num_channels)
            else:
                fourier = NonCartesianFFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape,
                    n_coils=self.num_channels)
            kspace_data = fourier.op(image_multichannel)
            linear_op, _ = \
                self.get_linear_n_regularization_operator(
                    wavelet_name=name,
                    dimension=len(fourier.shape),
                    nb_scale=2,
                    n_coils=2,
                    n_jobs=2,
                    image_shape=image.shape,
                )
            regularizer_op_gl = GroupLASSO(weights=0)
            linear_op.op(image_multichannel)
            regularizer_op_owl = OWL(
                alpha=0,
                beta=0,
                mode='band_based',
                n_coils=self.num_channels,
                bands_shape=linear_op.coeffs_shape,
            )
            for regularizer_op in [regularizer_op_gl, regularizer_op_owl]:
                reconstructor = CalibrationlessReconstructor(
                    fourier_op=fourier,
                    linear_op=linear_op,
                    regularizer_op=regularizer_op,
                    gradient_formulation=formulation,
                    num_check_lips=0,
                    verbose=1,
                )
                x_final, costs, _ = reconstructor.reconstruct(
                    kspace_data=kspace_data,
                    optimization_alg=optimizer,
                    num_iterations=10,
                )
                fourier_0 = FFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape,
                    n_coils=self.num_channels,
                )
                data_0 = fourier_0.op(image_multichannel)
                # mu is 0 for above single channel reconstruction and
                # hence we expect the result to be the inverse fourier
                # transform
                np.testing.assert_allclose(
                    x_final, fourier_0.adj_op(data_0), 0.01)

    def test_check_asserts(self):
        # Tests to check for asserts
        image, nb_scale, optimizer, recon_type, name = self.test_cases[0]
        fourier = NonCartesianFFT(
            samples=convert_mask_to_locations(self.mask),
            shape=image.shape,
        )
        kspace_data = fourier.op(image.data)
        linear_op, regularizer_op = \
            self.get_linear_n_regularization_operator(
                wavelet_name=name,
                dimension=len(fourier.shape),
                nb_scale=2,
                image_shape=image.shape,
            )
        reconstructor = CalibrationlessReconstructor(
            fourier_op=fourier,
            linear_op=linear_op,
            regularizer_op=regularizer_op,
            gradient_formulation="synthesis",
            verbose=1,
        )
        np.testing.assert_raises(
            ValueError,
            reconstructor.reconstruct,
            kspace_data=kspace_data,
            optimization_alg="test_fail",
            num_iterations=self.num_iter,
        )
        fourier.n_coils = 10
        reconstructor = SelfCalibrationReconstructor(
            fourier_op=fourier,
            linear_op=linear_op,
            regularizer_op=regularizer_op,
            gradient_formulation="synthesis",
            verbose=1,
        )
        np.testing.assert_raises(
            ValueError,
            reconstructor.reconstruct,
            kspace_data=kspace_data,
            optimization_alg=optimizer,
            num_iterations=self.num_iter,
        )

    def test_stack3d_self_calibration_recon(self):
        # This test carries out a self calibration recon using Stack3D
        self.num_channels = 2
        self.z_size = 10
        for i in range(len(self.test_cases)):
            image, nb_scale, optimizer, recon_type, name = self.test_cases[i]
            if recon_type == 'cartesian' or name == 24:
                continue
            # Make a dummy 3D image from 2D
            image = np.moveaxis(
                np.repeat(image.data[np.newaxis], self.z_size, axis=0), 0, 2)
            # Make dummy multichannel image
            image = np.repeat(image[np.newaxis], self.num_channels, axis=0)
            sampling_z = np.random.randint(2, size=image.shape[3])
            sampling_z[self.z_size//2-3:self.z_size//2+3] = 1
            Nz = sampling_z.sum()
            mask = convert_mask_to_locations(self.mask)
            z_locations = np.repeat(convert_mask_to_locations(sampling_z),
                                    mask.shape[0])
            z_locations = z_locations[:, np.newaxis]
            kspace_loc = np.hstack([np.tile(mask, (Nz, 1)), z_locations])
            fourier = Stacked3DNFFT(kspace_loc=kspace_loc,
                                    shape=image.shape[1:],
                                    implementation='cpu',
                                    n_coils=self.num_channels)
            kspace_obs = fourier.op(image)
            if optimizer == 'condatvu':
                formulation = "analysis"
            else:
                formulation = "synthesis"
            linear_op, regularizer_op = \
                self.get_linear_n_regularization_operator(
                    wavelet_name=name,
                    dimension=len(fourier.shape),
                    nb_scale=2,
                    n_coils=1,
                    n_jobs=2,
                    image_shape=image.shape[1:],
                )
            reconstructor = SelfCalibrationReconstructor(
                fourier_op=fourier,
                linear_op=linear_op,
                regularizer_op=regularizer_op,
                gradient_formulation=formulation,
                num_check_lips=0,
                smaps_extraction_mode='Stack',
                verbose=1,
            )
            x_final, _, _, = reconstructor.reconstruct(
                kspace_data=kspace_obs,
                optimization_alg=optimizer,
                num_iterations=5,
            )
            fourier_0 = FFT(
                samples=kspace_loc,
                shape=image.shape[1:],
                n_coils=self.num_channels,
            )
            recon = fourier_0.adj_op(fourier_0.op(image))
            np.testing.assert_allclose(
                np.abs(x_final),
                np.sqrt(np.sum(np.abs(recon)**2, axis=0)), 0.1)

    def test_online_accumulating_calibrationless(self):
        self.num_channels = 2
        for i in range(len(self.test_cases)):
            image, nb_scale, optimizer, recon_type, name = self.test_cases[i]
            if recon_type != 'cartesian':
                continue
            if optimizer == 'condatvu':
                formulation = "analysis"
            else:
                formulation = "synthesis"
            image_multichannel = np.repeat(image.data[np.newaxis],
                                           self.num_channels, axis=0)

            fourier = FFT(
                samples=convert_mask_to_locations(self.mask),
                shape=image.shape,
                n_coils=self.num_channels)
            kspace_data = fourier.op(image_multichannel)

            linear_op, _ = \
                self.get_linear_n_regularization_operator(
                    wavelet_name=name,
                    dimension=len(fourier.shape),
                    nb_scale=2,
                    n_coils=2,
                    n_jobs=2,
                    image_shape=image.shape,
                )
            regularizer_op_gl = GroupLASSO(weights=0)
            linear_op.op(image_multichannel)
            regularizer_op_owl = OWL(
                alpha=0,
                beta=0,
                mode='band_based',
                n_coils=self.num_channels,
                bands_shape=linear_op.coeffs_shape,
            )
            for regularizer_op in [regularizer_op_gl, regularizer_op_owl]:
                print(image, nb_scale, optimizer, recon_type, name, regularizer_op)
                kspace_gen = KspaceGeneratorBase(full_kspace=kspace_data, mask=fourier.mask, max_iter=10)
                reconstructor = CalibrationlessReconstructor(
                    fourier_op=fourier,
                    linear_op=linear_op,
                    regularizer_op=regularizer_op,
                    gradient_formulation=formulation,
                    num_check_lips=0,
                    verbose=1,
                )
                x_final, costs, _ = reconstructor.reconstruct(
                    kspace_data=kspace_gen,
                    optimization_alg=optimizer,
                )
                fourier_0 = FFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape,
                    n_coils=self.num_channels,
                )
                data_0 = fourier_0.op(image_multichannel)
                # mu is 0 for above single channel reconstruction and
                # hence we expect the result to be the inverse fourier
                # transform
                np.testing.assert_allclose(
                    x_final, fourier_0.adj_op(data_0), 0.01)


if __name__ == "__main__":
    unittest.main()
