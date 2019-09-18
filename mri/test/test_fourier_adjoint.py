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
import time

# Package import
from mri.reconstruct.fourier import FFT2, NFFT, Stacked3D
from mri.reconstruct.utils import convert_mask_to_locations
from mri.reconstruct.utils import convert_locations_to_mask
from mri.reconstruct.utils import normalize_frequency_locations


class TestAdjointOperatorFourierTransform(unittest.TestCase):
    """ Test the adjoint operator of the NFFT both for 2D and 3D.
    """
    def setUp(self):
        """ Set the number of iterations.
        """
        self.N = 64
        self.max_iter = 10

    def test_normalize_frequency_locations_2D(self):
        """Test the output of the normalize frequency methods and check that it
        is indeed between [-0.5; 0.5[
        """
        for _ in range(10):
            samples = np.random.randn(128*128, 2)
            normalized_samples = normalize_frequency_locations(samples)
            self.assertFalse((normalized_samples.all() < 0.5 and
                             normalized_samples.all() >= -0.5))
        print(" Test normalization function for 2D input passes")

    def test_normalize_frequency_locations_3D(self):
        """Test the output of the normalize frequency methods and check that it
        is indeed between [-0.5; 0.5[
        """
        for _ in range(10):
            samples = np.random.randn(128*128, 3)
            normalized_samples = normalize_frequency_locations(samples)
            self.assertFalse((normalized_samples.all() < 0.5 and
                             normalized_samples.all() >= -0.5))
        print(" Test normalization function for 3D input passes")

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

    def test_FFT2(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            _mask = np.random.randint(2, size=(self.N, self.N))
            _samples = convert_mask_to_locations(_mask)
            print("Process FFT2 test '{0}'...", i)
            fourier_op_dir = FFT2(samples=_samples, shape=(self.N, self.N))
            fourier_op_adj = FFT2(samples=_samples, shape=(self.N, self.N))
            Img = (np.random.randn(self.N, self.N) +
                   1j * np.random.randn(self.N, self.N))
            f = (np.random.randn(self.N, self.N) +
                 1j * np.random.randn(self.N, self.N))
            f_p = fourier_op_dir.op(Img)
            I_p = fourier_op_adj.adj_op(f)
            x_d = np.vdot(Img, I_p)
            x_ad = np.vdot(f_p, f)
            np.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print(" FFT2 adjoint test passes")

    def test_NFFT_2D(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            _mask = np.random.randint(2, size=(self.N, self.N))
            _samples = convert_mask_to_locations(_mask)
            print("Process NFFT in 2D test '{0}'...", i)
            fourier_op_dir = NFFT(samples=_samples, shape=(self.N, self.N))
            fourier_op_adj = NFFT(samples=_samples, shape=(self.N, self.N))
            Img = np.random.randn(self.N, self.N) + \
                1j * np.random.randn(self.N, self.N)
            f = np.random.randn(_samples.shape[0], 1) + \
                1j * np.random.randn(_samples.shape[0], 1)
            f_p = fourier_op_dir.op(Img)
            I_p = fourier_op_adj.adj_op(f)
            x_d = np.vdot(Img, I_p)
            x_ad = np.vdot(f_p, f)
            np.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print(" NFFT in 2D adjoint test passes")

    def test_NFFT_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            _mask = np.random.randint(2, size=(self.N, self.N, self.N))
            _samples = convert_mask_to_locations(_mask)
            print("Process NFFT test in 3D '{0}'...", i)
            fourier_op_dir = NFFT(samples=_samples,
                                  shape=(self.N, self.N, self.N))
            fourier_op_adj = NFFT(samples=_samples,
                                  shape=(self.N, self.N, self.N))
            Img = np.random.randn(self.N, self.N, self.N) + \
                1j * np.random.randn(self.N, self.N, self.N)
            f = np.random.randn(_samples.shape[0], 1) + \
                1j * np.random.randn(_samples.shape[0], 1)
            f_p = fourier_op_dir.op(Img)
            I_p = fourier_op_adj.adj_op(f)
            x_d = np.vdot(Img, I_p)
            x_ad = np.vdot(f_p, f)
            np.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print(" NFFT in 3D adjoint test passes")

    def test_adjoint_stack_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            _mask = np.random.randint(2, size=(self.N, self.N))
            _mask3D = np.asarray([_mask for i in np.arange(self.N)])
            _samples = convert_mask_to_locations(_mask3D.swapaxes(0, 2))
            print("Process Stacked3D-FFT test in 3D '{0}'...", i)
            fourier_op = Stacked3D(samples=_samples,
                                   shape=(self.N, self.N, self.N))
            Img = (np.random.random((self.N, self.N, self.N))
                   + 1j * np.random.random((self.N, self.N, self.N)))
            f = (np.random.random((_samples.shape[0], 1))
                 + 1j * np.random.random((_samples.shape[0], 1)))
            f_p = fourier_op.op(Img)
            I_p = fourier_op.adj_op(f)
            x_d = np.vdot(Img, I_p)
            x_ad = np.vdot(f_p, f)
            np.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print("Stacked FFT in 3D adjoint test passes")

    def test_similarity_stack_3D(self):
        """Test the similarity of stacked implementation of Fourier transform
        to that of NFFT
        """
        for N in [64, 128]:
            # Nz is the number of slices, this would check both N=Nz and N!=Nz
            Nz = 64
            _mask = np.random.randint(2, size=(N, N))
            _mask3D = np.asarray([_mask for i in np.arange(Nz)])
            _samples = convert_mask_to_locations(_mask3D.swapaxes(0, 2))
            print("Process Stack-3D similarity with NFFT for N=" + str(N))
            fourier_op_stack = Stacked3D(samples=_samples,
                                         shape=(N, N, Nz))
            fourier_op_nfft = NFFT(samples=_samples,
                                   shape=(N, N, Nz))
            Img = (np.random.random((N, N, Nz))
                   + 1j * np.random.random((N, N, Nz)))
            f = (np.random.random((_samples.shape[0], 1))
                 + 1j * np.random.random((_samples.shape[0], 1)))
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
                  str(nfft_runtime/stack_runtime))
        print("Stacked FFT in 3D adjoint test passes")


if __name__ == "__main__":
    unittest.main()
