# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import unittest
import numpy as np

# Package import
from mri.operators import NonCartesianFFT
from mri.operators.utils import convert_mask_to_locations
from mri.reconstructors.utils.extract_sensitivity_maps \
    import get_Smaps, extract_k_space_center_and_locations


class TestSensitivityExtraction(unittest.TestCase):
    """ Test the code for sensitivity extraction
    """

    def setUp(self):
        """ Initialization of variables for tests:
            N = Image size in 2D
            Nz = Number of slices
            num_channel = Number of channels
            percent = percent of kspace to extract while testing
        """
        self.N = 64
        self.Nz = 60
        self.num_channel = 3
        # Percent of k-space center
        self.percent = 0.5

    def test_extract_k_space_center_3D(self):
        """ This test ensures that the output of the non cartesian kspace
        extraction is same a that of mimicked cartesian extraction in 3D
        """
        _mask = np.ones((self.N, self.N, self.Nz))
        _samples = convert_mask_to_locations(_mask)
        Img = (np.random.randn(self.num_channel, self.N, self.N, self.Nz) +
               1j * np.random.randn(self.num_channel, self.N, self.N,
                                    self.Nz))
        Nby2_percent = self.N * self.percent / 2
        Nzby2_percent = self.Nz * self.percent / 2
        low = int(self.N / 2 - Nby2_percent)
        high = int(self.N / 2 + Nby2_percent + 1)
        lowz = int(self.Nz / 2 - Nzby2_percent)
        highz = int(self.Nz / 2 + Nzby2_percent + 1)
        center_Img = Img[:, low:high, low:high, lowz:highz]
        thresh = self.percent * 0.5
        data_thresholded, samples_thresholded = \
            extract_k_space_center_and_locations(
                data_values=np.reshape(Img, (self.num_channel,
                                             self.N * self.N * self.Nz)),
                samples_locations=_samples,
                thr=(thresh, thresh, thresh),
                img_shape=(self.N, self.N, self.Nz))
        np.testing.assert_allclose(
            center_Img.reshape(data_thresholded.shape),
            data_thresholded)

    def test_extract_k_space_center_2D(self):
        """ Ensure that the extracted k-space center is right,
        send density compensation also and vet the code path"""
        mask = np.ones((self.N, self.N))
        samples = convert_mask_to_locations(mask)
        Img = (np.random.randn(self.num_channel, self.N, self.N) +
               1j * np.random.randn(self.num_channel, self.N, self.N))
        Nby2_percent = self.N * self.percent / 2
        low = int(self.N / 2 - Nby2_percent)
        high = int(self.N / 2 + Nby2_percent + 1)
        center_Img = Img[:, low:high, low:high]
        thresh = self.percent * 0.5
        data_thresholded, samples_thresholded, dc = \
            extract_k_space_center_and_locations(
                data_values=np.reshape(Img, (self.num_channel,
                                             self.N * self.N)),
                samples_locations=samples,
                thr=(thresh, thresh),
                img_shape=(self.N, self.N),
                density_comp=np.ones(samples.shape[0])
            )
        np.testing.assert_allclose(
            center_Img.reshape(data_thresholded.shape),
            data_thresholded)

    def test_extract_k_space_center_2D_fft(self):
        """ Ensure that the extracted k-space center is right
        for cartesian case"""
        mask = np.random.randint(0, 2, (self.N, self.N))
        samples = convert_mask_to_locations(mask)
        Img = (np.random.randn(self.num_channel, self.N, self.N) +
               1j * np.random.randn(self.num_channel, self.N, self.N))
        Nby2_percent = self.N * self.percent / 2
        low = int(self.N / 2 - Nby2_percent)
        high = int(self.N / 2 + Nby2_percent + 1)
        center_Img = Img[:, low:high, low:high]
        cutoff_mask = mask[low:high, low:high]
        locations = np.where(cutoff_mask.reshape(cutoff_mask.size))
        center_Img = center_Img.reshape(
            (center_Img.shape[0], cutoff_mask.size))[:, locations[0]]
        thresh = self.percent * 0.5
        data_thresholded, samples_thresholded = \
            extract_k_space_center_and_locations(
                data_values=Img,
                samples_locations=samples,
                thr=(thresh, thresh),
                img_shape=(self.N, self.N))
        np.testing.assert_allclose(
            center_Img,
            data_thresholded)

    def test_sensitivity_extraction_2D(self):
        """ This test ensures that the output of the non cartesian kspace
        extraction is same a that of mimicked cartesian extraction in 2D
        """
        mask = np.ones((self.N, self.N))
        samples = convert_mask_to_locations(mask)
        fourier_op = NonCartesianFFT(samples=samples, shape=(self.N, self.N))
        Img = (np.random.randn(self.num_channel, self.N, self.N) +
               1j * np.random.randn(self.num_channel, self.N, self.N))
        F_img = np.asarray([fourier_op.op(Img[i])
                            for i in np.arange(self.num_channel)])
        Smaps_gridding, SOS_Smaps = get_Smaps(
            k_space=F_img,
            img_shape=(self.N, self.N),
            samples=samples,
            thresh=(0.4, 0.4),
            mode='gridding',
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            n_cpu=1)
        Smaps_NFFT_dc, SOS_Smaps_dc = get_Smaps(
            k_space=F_img,
            img_shape=(self.N, self.N),
            thresh=(0.4, 0.4),
            samples=samples,
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            mode='NFFT',
            density_comp=np.ones(samples.shape[0])
        )
        Smaps_NFFT, SOS_Smaps = get_Smaps(
            k_space=F_img,
            img_shape=(self.N, self.N),
            thresh=(0.4, 0.4),
            samples=samples,
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            mode='NFFT',
        )
        Smaps_hann_NFFT, SOS_Smaps = get_Smaps(
            k_space=F_img,
            img_shape=(self.N, self.N),
            thresh=0.4,
            samples=samples,
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            window_fun="Hann",
            mode='NFFT',
        )
        Smaps_hann_gridding, SOS_Smaps = get_Smaps(
            k_space=F_img,
            img_shape=(self.N, self.N),
            thresh=0.4,
            samples=samples,
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            window_fun="Hann",
            mode='gridding',
        )

        Smaps_hamming_NFFT, SOS_Smaps = get_Smaps(
            k_space=F_img,
            img_shape=(self.N, self.N),
            thresh=0.4,
            samples=samples,
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            window_fun="Hamming",
            mode='NFFT',
        )
        Smaps_hamming_gridding, SOS_Smaps = get_Smaps(
            k_space=F_img,
            img_shape=(self.N, self.N),
            thresh=0.4,
            samples=samples,
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            window_fun="Hamming",
            mode='gridding',
        )
        Smaps_call_gridding, SOS_Smaps = get_Smaps(
            k_space=F_img,
            img_shape=(self.N, self.N),
            thresh=0.4,
            samples=samples,
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            window_fun=lambda x: 1,
            mode='gridding',
        )
        Smaps_call_NFFT, SOS_Smaps = get_Smaps(
            k_space=F_img,
            img_shape=(self.N, self.N),
            thresh=0.4,
            samples=samples,
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            window_fun=lambda x: 1,
            mode='NFFT',
        )

        np.testing.assert_allclose(Smaps_gridding, Smaps_NFFT_dc)
        np.testing.assert_allclose(Smaps_gridding, Smaps_NFFT)
        np.testing.assert_allclose(Smaps_hann_gridding, Smaps_hann_NFFT)
        np.testing.assert_allclose(Smaps_hamming_gridding, Smaps_hamming_NFFT)
        np.testing.assert_allclose(Smaps_call_gridding, Smaps_call_NFFT)
        # Test that we raise assert for bad mode
        np.testing.assert_raises(
            ValueError,
            get_Smaps,
            k_space=F_img,
            img_shape=(self.N, self.N),
            thresh=(0.4, 0.4),
            samples=samples,
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            mode='test'
        )
        # Test that we raise assert for bad window
        np.testing.assert_raises(
            ValueError,
            get_Smaps,
            k_space=F_img,
            img_shape=(self.N, self.N),
            thresh=(0.4, 0.4),
            samples=samples,
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            window_fun='test',
            mode='gridding',
        )

if __name__ == "__main__":
    unittest.main()
