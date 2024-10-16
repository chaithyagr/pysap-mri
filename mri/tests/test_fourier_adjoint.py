# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import unittest
import numpy as np
from itertools import product

# Package import
from mri.operators import FFT, NonCartesianFFT, Stacked3DNFFT
from mri.operators.utils import convert_mask_to_locations, \
    convert_locations_to_mask, normalize_frequency_locations, \
    discard_frequency_outliers, get_stacks_fourier
import time


class TestAdjointOperatorFourierTransform(unittest.TestCase):
    """ Test the adjoint operator of the Fourier in both for 2D and 3D.
    """
    def setUp(self):
        """ Set the number of iterations.
        """
        self.N = 64
        self.max_iter = 10
        self.num_channels = [1, 2]

    def test_normalize_frequency_locations_2D(self):
        """Test the output of the normalize frequency methods and check that it
        is indeed within [-0.5; 0.5[
        """
        for _ in range(10):
            samples = np.random.randn(128*128, 2)
            normalized_samples = normalize_frequency_locations(samples)
            self.assertTrue((normalized_samples < 0.5).all() and
                            (normalized_samples >= -0.5).all())
        print(" Test normalization function for 2D input passes")

    def test_normalize_frequency_locations_3D(self):
        """Test the output of the normalize frequency methods and check that it
        is indeed within [-0.5; 0.5[
        """
        for _ in range(10):
            samples = np.random.randn(128*128, 3)
            normalized_samples = normalize_frequency_locations(samples)
            self.assertTrue((normalized_samples < 0.5).all() and
                            (normalized_samples >= -0.5).all())
        print(" Test normalization function for 3D input passes")

    def test_discard_frequency_outliers_2D(self):
        """Test the output of the discard frequency methods, checking that
        locations are within [-0.5; 0.5[ and that locations and samples are
        similarly discarded.
        """
        for _ in range(10):
            kspace_loc = np.random.randn(128*128, 2)
            kspace_data = np.random.randn(1, 128*128, 2)
            reduced_loc, reduced_data = discard_frequency_outliers(kspace_loc,
                                                                   kspace_data)
            self.assertTrue((reduced_loc < 0.5).all() and
                            (reduced_loc >= -0.5).all())
            self.assertEqual(reduced_loc.shape[0], reduced_data.shape[1])
        print(" Test location discarding function for 2D input passes")

    def test_discard_frequency_outliers_3D(self):
        """Test the output of the discard frequency methods, checking that
        locations are within [-0.5; 0.5[ and that locations and samples are
        similarly discarded.
        """
        for _ in range(10):
            kspace_loc = np.random.randn(128*128, 3)
            kspace_data = np.random.randn(1, 128*128, 3)
            reduced_loc, reduced_data = discard_frequency_outliers(kspace_loc,
                                                                   kspace_data)
            self.assertTrue((reduced_loc < 0.5).all() and
                            (reduced_loc >= -0.5).all())
            self.assertEqual(reduced_loc.shape[0], reduced_data.shape[1])
        print(" Test location discarding function for 3D input passes")

    def test_sampling_converters(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            print("Process test convert mask to samples test '{0}'...", i)
            Nx = np.random.randint(8, 512)
            Ny = np.random.randint(8, 512)
            mask = np.random.randint(2, size=(Nx, Ny))
            samples = convert_mask_to_locations(mask)
            recovered_mask = convert_locations_to_mask(samples,
                                                       (Nx, Ny))
            self.assertEqual(mask.all(), recovered_mask.all())
            mismatch = 0. + (np.mean(
                np.allclose(mask, recovered_mask)))
            print("      mismatch = ", mismatch)
        print(" Test convert mask to samples and it's adjoint passes for",
              " the 2D cases")

    def test_sampling_converters_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier
        transform
        """
        for i in range(self.max_iter):
            print("Process test convert mask to samples test '{0}'...", i)
            Nx = np.random.randint(8, 512)
            Ny = np.random.randint(8, 512)
            Nz = np.random.randint(8, 512)
            mask = np.random.randint(2, size=(Nx, Ny, Nz))
            samples = convert_mask_to_locations(mask)
            recovered_mask = convert_locations_to_mask(samples,
                                                       (Nx, Ny, Nz))
            self.assertEqual(mask.all(), recovered_mask.all())
            mismatch = 0. + (np.mean(
                np.allclose(mask, recovered_mask)))
            print("      mismatch = ", mismatch)
        print(" Test convert mask to samples and it's adjoint passes for",
              " the 3D cases")

    def test_FFT(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        """
        for i, num_coil in product(range(self.max_iter), self.num_channels):
            _mask = np.random.randint(2, size=(self.N, self.N))
            _samples = convert_mask_to_locations(_mask)
            print("Process FFT test '{0}'...", i)
            fourier_op_dir = FFT(samples=_samples, shape=(self.N, self.N),
                                 n_coils=num_coil)
            fourier_op_adj = FFT(samples=_samples, shape=(self.N, self.N),
                                 n_coils=num_coil)
            Img = np.squeeze(np.random.randn(num_coil, self.N, self.N) +
                             1j * np.random.randn(num_coil, self.N, self.N))
            f = np.squeeze(np.random.randn(num_coil, self.N, self.N) +
                           1j * np.random.randn(num_coil, self.N, self.N))
            f_p = fourier_op_dir.op(Img)
            I_p = fourier_op_adj.adj_op(f)
            x_d = np.vdot(Img, I_p)
            x_ad = np.vdot(f_p, f)
            np.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print(" FFT adjoint test passes")

    def test_NFFT_2D(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier
        transform, with density compensator set to 1, to vet the code
        path, the test is unchanged otherwise
        """
        for num_channels in self.num_channels:
            print("Testing with num_channels=" + str(num_channels))
            for i in range(self.max_iter):
                _mask = np.random.randint(2, size=(self.N, self.N))
                _samples = convert_mask_to_locations(_mask)
                print("Process NFFT in 2D test '{0}'...", i)
                fourier_op = NonCartesianFFT(
                    samples=_samples,
                    shape=(self.N, self.N),
                    n_coils=num_channels,
                    implementation='cpu',
                    density_comp=np.ones((num_channels, _samples.shape[0]))
                )
                Img = np.random.randn(num_channels, self.N, self.N) + \
                    1j * np.random.randn(num_channels, self.N, self.N)
                f = np.random.randn(num_channels, _samples.shape[0]) + \
                    1j * np.random.randn(num_channels, _samples.shape[0])
                f_p = fourier_op.op(Img)
                I_p = fourier_op.adj_op(f)
                x_d = np.vdot(Img, I_p)
                x_ad = np.vdot(f_p, f)
                np.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print(" NFFT in 2D adjoint test passes")

    def test_NFFT_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier transform
        """
        for num_channels in self.num_channels:
            print("Testing with num_channels=" + str(num_channels))
            for i in range(self.max_iter):
                _mask = np.random.randint(2, size=(self.N, self.N, self.N))
                _samples = convert_mask_to_locations(_mask)
                print("Process NFFT test in 3D '{0}'...", i)
                fourier_op = NonCartesianFFT(samples=_samples,
                                             shape=(self.N, self.N, self.N),
                                             n_coils=num_channels,
                                             implementation='cpu')
                Img = np.random.randn(num_channels, self.N, self.N, self.N) + \
                    1j * np.random.randn(num_channels, self.N, self.N, self.N)
                f = np.random.randn(num_channels, _samples.shape[0]) + \
                    1j * np.random.randn(num_channels, _samples.shape[0])
                f_p = fourier_op.op(Img)
                I_p = fourier_op.adj_op(f)
                x_d = np.vdot(Img, I_p)
                x_ad = np.vdot(f_p, f)
                np.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print(" NFFT in 3D adjoint test passes")

    def test_adjoint_stack_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier transform
        """
        for channel in self.num_channels:
            print("Testing with num_channels=" + str(channel))
            for i in range(self.max_iter):
                mask = np.random.randint(2, size=(self.N, self.N))
                sampling_z = np.random.randint(2, size=self.N)
                sampling_z[self.N//2-10:self.N//2+10] = 1
                Nz = sampling_z.sum()
                mask = convert_mask_to_locations(mask)
                z_locations = np.repeat(
                    convert_mask_to_locations(sampling_z),
                    mask.shape[0],
                )
                z_locations = z_locations[:, np.newaxis]
                kspace_loc = np.hstack([np.tile(mask, (Nz, 1)), z_locations])
                print("Process Stacked3D-FFT test in 3D '{0}'...", i)
                fourier_op = Stacked3DNFFT(
                    kspace_loc=kspace_loc,
                    shape=(self.N, self.N, self.N),
                    implementation='cpu',
                    n_coils=channel,
                )
                Img = np.random.random((channel, self.N, self.N, self.N)) + \
                    1j * np.random.random((channel, self.N, self.N, self.N))
                f = np.random.random((channel, kspace_loc.shape[0])) + \
                    1j * np.random.random((channel, kspace_loc.shape[0]))
                f_p = fourier_op.op(Img)
                I_p = fourier_op.adj_op(f)
                x_d = np.vdot(Img, I_p)
                x_ad = np.vdot(f_p, f)
                np.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print("Stacked FFT in 3D adjoint test passes")

    def test_similarity_stack_3D_nfft(self):
        """Test the similarity of stacked implementation of Fourier transform
        to that of NFFT
        """
        for channel in self.num_channels:
            print("Testing with num_channels=" + str(channel))
            for N in [16, 32]:
                # Nz is the number of slices, this would check both N=Nz
                # and N!=Nz
                Nz = 16
                _mask = np.random.randint(2, size=(N, N))

                # Generate random mask along z
                sampling_z = np.random.randint(2, size=Nz)
                _mask3D = np.zeros((N, N, Nz))
                for idx, acq_z in enumerate(sampling_z):
                    _mask3D[:, :, idx] = _mask * acq_z
                _samples = convert_mask_to_locations(_mask3D)

                print("Process Stack-3D similarity with NFFT for N=" + str(N))
                fourier_op_stack = Stacked3DNFFT(kspace_loc=_samples,
                                                 shape=(N, N, Nz),
                                                 implementation='cpu',
                                                 n_coils=channel)
                fourier_op_nfft = NonCartesianFFT(samples=_samples,
                                                  shape=(N, N, Nz),
                                                  implementation='cpu',
                                                  n_coils=channel)
                Img = np.random.random((channel, N, N, Nz)) + \
                    1j * np.random.random((channel, N, N, Nz))
                f = np.random.random((channel, _samples.shape[0])) + \
                    1j * np.random.random((channel, _samples.shape[0]))
                start_time = time.time()
                stack_f_p = fourier_op_stack.op(Img)
                stack_I_p = fourier_op_stack.adj_op(f)
                stack_runtime = time.time() - start_time
                start_time = time.time()
                nfft_f_p = fourier_op_nfft.op(Img)
                nfft_I_p = fourier_op_nfft.adj_op(f)
                np.testing.assert_allclose(stack_f_p, nfft_f_p, rtol=1e-9)
                np.testing.assert_allclose(stack_I_p, nfft_I_p, rtol=1e-9)
                nfft_runtime = time.time() - start_time
                print("For N=" + str(N) + " Speedup = " +
                      str(nfft_runtime / stack_runtime))
        print("Stacked FFT in 3D adjoint test passes")

    def test_stack_3D_error(self):
        np.testing.assert_raises(ValueError,
                                 get_stacks_fourier, np.random.randn(12, 3),
                                 (self.N, self.N, self.N))
        # Generate random mask along z
        sampling_z = np.random.randint(2, size=self.N)
        _mask3D = np.zeros((self.N, self.N, self.N))
        for idx, acq_z in enumerate(sampling_z):
            _mask3D[:, :, idx] = np.random.randint(
                2,
                size=(self.N, self.N)) * acq_z
        sampling = convert_mask_to_locations(_mask3D)
        np.testing.assert_raises(ValueError,
                                 get_stacks_fourier, sampling,
                                 (self.N, self.N, self.N))


if __name__ == "__main__":
    unittest.main()
