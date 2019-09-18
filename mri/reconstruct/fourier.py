# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Fourier operators for cartesian and non-cartesian space.
"""

# System import
import warnings
import numpy as np

# Package import
from .utils import convert_locations_to_mask
from .utils import normalize_frequency_locations

# Third party import
try:
    import pynfft
except Exception:
    warnings.warn("pynfft python package has not been found. If needed use "
                  "the master release.")
    pass


class FourierBase(object):
    """ Base Fourier transform operator class.
    """
    def op(self, img):
        """ This method calculates Fourier transform.
        Parameters
        ----------
        img: np.ndarray
            input image as array.

        Returns
        -------
        result: np.ndarray
            Fourier transform of the image.
        """
        raise NotImplementedError("'op' is an abstract method.")

    def adj_op(self, x):
        """ This method calculates inverse Fourier transform of real or complex
        sequence.

        Parameters
        ----------
        x: np.ndarray
            input Fourier data array.

        Returns
        -------
        results: np.ndarray
            inverse discrete Fourier transform.
        """
        raise NotImplementedError("'adj_op' is an abstract method.")


class FFT2(FourierBase):
    """ Standard unitary 2D Fast Fourrier Transform class.
    The FFT2 will be normalized in a symmetric way

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    """
    def __init__(self, samples, shape):
        """ Initilize the 'FFT2' class.

        Parameters
        ----------
        samples: np.ndarray
            the mask samples in the Fourier domain.
        shape: tuple of int
            shape of the image (not necessarly a square matrix).
        """
        self.samples = samples
        self.shape = shape
        self._mask = convert_locations_to_mask(self.samples, self.shape)

    def op(self, img):
        """ This method calculates the masked Fourier transform of a 2-D image.

        Parameters
        ----------
        img: np.ndarray
            input 2D array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        return self._mask * np.fft.fft2(img, norm="ortho")

    def adj_op(self, x):
        """ This method calculates inverse masked Fourier transform of a 2-D
        image.

        Parameters
        ----------
        x: np.ndarray
            masked Fourier transform data.

        Returns
        -------
        img: np.ndarray
            inverse 2D discrete Fourier transform of the input coefficients.
        """
        return np.fft.ifft2(self._mask * x, norm="ortho")


class NFFT(FourierBase):
    """ ND non catesian Fast Fourrier Transform class
    The NFFT will normalize like the FFT2 i.e. in a symetric way.
    This means that both direct and adjoint operator will be divided by the
    square root of the number of samples in the fourier domain.

    Attributes
    ----------
    samples: np.ndarray
        the samples locations in the Fourier domain between [-0.5; 0.5[.
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    """

    def __init__(self, samples, shape):
        """ Initilize the 'NFFT' class.

        Parameters
        ----------
        samples: np.ndarray (Mxd)
            the samples locations in the Fourier domain where M is the number
            of samples and d is the dimensionnality of the output data
            (2D for an image, 3D for a volume).
        shape: tuple of int
            shape of the image (not necessarly a square matrix).

        Exemple
        -------
        >>> import numpy as np
        >>> from pysap.data import get_sample_data
        >>> from mri.numerics.fourier import NFFT, FFT2
        >>> from mri.reconstruct.utils import \
        convert_mask_to_locations

        >>> I = get_sample_data("2d-pmri").data.astype("complex128")
        >>> I = I[0]
        >>> samples = convert_mask_to_locations(np.ones(I.shape))
        >>> fourier_op = NFFT(samples=samples, shape=I.shape)
        >>> cartesian_fourier_op = FFT2(samples=samples, shape=I.shape)
        >>> x_nfft = fourier_op.op(I)
        >>> x_fft = np.fft.ifftshift(cartesian_fourier_op.op(
                np.fft.fftshift(I))).flatten()
        >>> np.mean(np.abs(x_fft / x_nfft))
        1.000000000000005
        """
        if samples.shape[-1] != len(shape):
            raise ValueError("Samples and Shape dimension doesn't correspond")
        self.samples = samples
        if samples.min() < -0.5 or samples.max() >= 0.5:
            warnings.warn("Samples will be normalized between [-0.5; 0.5[")
            self.samples = normalize_frequency_locations(self.samples)
        self.plan = pynfft.NFFT(N=shape, M=len(samples))
        self.shape = shape
        self.plan.x = self.samples
        self.plan.precompute()

    def op(self, img):
        """ This method calculates the masked non-cartesian Fourier transform
        of a N-D data.

        Parameters
        ----------
        img: np.ndarray
            input ND array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        self.plan.f_hat = img
        return np.copy(self.plan.trafo()) / np.sqrt(self.plan.M)

    def adj_op(self, x):
        """ This method calculates inverse masked non-cartesian Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        x: np.ndarray
            masked non-cartesian Fourier transform 1D data.

        Returns
        -------
        img: np.ndarray
            inverse 2D discrete Fourier transform of the input coefficients.
        """
        self.plan.f = x
        return np.copy(self.plan.adjoint()) / np.sqrt(self.plan.M)


class Stacked3D:
    """"  3-D non uniform Fast Fourrier Transform class,
        fast implementation for Stacked samples
    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (necessarly a square/cubic matrix).
    fft_type: 'NFFT' | 'NUFFT' default 'NFFT'
        What FFT version to be used for 2D non uniform FFT
    platform: string, 'cpu', 'multi-cpu' or 'gpu' default 'gpu'
        string indicating which hardware platform will be used to compute the
        NUFFT. works only if fft_type=='NUFFT'
    """

    def __init__(self, samples, shape, fft_type='NFFT', platform='gpu'):
        """ Init function for Stacked3D class
        Parameters
        ----------
        samples: np.ndarray
            the mask samples in the Fourier domain.
        shape: tuple of int
            shape of the image (necessarly a square/cubic matrix).
        fft_type: 'NFFT' | 'NUFFT' default 'NFFT'
            What FFT version to be used for 2D non uniform FFT
        platform: string, 'cpu', 'multi-cpu' or 'gpu' default 'gpu'
            string indicating which hardware platform will be used to
            compute the NUFFT. works only if fft_type=='NUFFT'
        """
        self.num_slices = shape[2]
        self.shape = shape
        # Sort the incoming data based on Z, Y then X coordinates
        # This is done for easier stacking
        self.sort_pos = np.lexsort(tuple(samples[:, i]
                                         for i in np.arange(3)))
        samples = samples[self.sort_pos]
        plane_samples, self.z_samples = self.get_stacks(samples)
        if fft_type == 'NUFFT':
            self.FT = NUFFT(samples=plane_samples, shape=shape[0:2],
                            platform=platform)
        elif fft_type == 'NFFT':
            self.FT = NFFT(samples=plane_samples, shape=shape[0:2])

    def get_stacks(self, samples):
        """ Function that converts an incoming 3D kspace samples
            and converts to stacks of 2D. This function also checks for
            any issues incoming k-space pattern and if the stack property
            is not satisfied.
            Stack Property:
                The k-space locations originate from a stack of 2D samples
        Parameters
        ----------
        samples: np.ndarray
            the mask samples in the 3D Fourier domain.

        Returns
        ----------
        plane_samples: np.ndarray
            A 2D array of samples which when stacked gives the 3D samples
        z_samples: np.ndarray
            A 1D array of z-sample locations
        """
        self.first_stack_len = np.size(np.where(samples[:, 2]
                                                == np.min(samples[:, 2])))
        self.acq_num_slices = int(len(samples) / self.first_stack_len)
        stacked = np.reshape(samples, (self.acq_num_slices,
                                       self.first_stack_len, 3))
        z_expected_stacked = np.reshape(np.repeat(stacked[:, 0, 2],
                                                  self.first_stack_len),
                                        (self.acq_num_slices,
                                         self.first_stack_len))
        if np.mod(len(samples), self.first_stack_len) \
                or not np.all(stacked[:, :, 0:2] == stacked[0, :, 0:2]) \
                or not np.all(stacked[:, :, 2] == z_expected_stacked):
            raise ValueError('The input must be a stack of 2D k-Space data')
        plane_samples = stacked[0, :, 0:2]
        z_samples = stacked[:, 0, 2]
        z_samples = z_samples[:, np.newaxis]
        return plane_samples, z_samples

    def op(self, data):
        """ This method calculates Fourier transform.
        Parameters
        ----------
        img: np.ndarray
            input image as array.

        Returns
        -------
        result: np.ndarray
            Forward 3D Fourier transform of the image.
        """
        first_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data, axes=2),
                                               n=self.acq_num_slices,
                                               norm="ortho"),
                                    axes=2)
        final = np.asarray([self.FT.op(first_fft[:, :, slice])
                            for slice in np.arange(self.acq_num_slices)])
        final = np.reshape(final, self.acq_num_slices * self.first_stack_len)
        # Unsort the Coefficients and send
        inv_idx = np.zeros_like(self.sort_pos)
        inv_idx[self.sort_pos] = np.arange(len(self.sort_pos))
        return final[inv_idx]

    def adj_op(self, coeff):
        """ This method calculates inverse masked non-uniform Fourier
        transform of a 1-D coefficients array.
        Parameters
        ----------
        x: np.ndarray
            masked non-uniform Fourier transform 1D data.
        Returns
        -------
        img: np.ndarray
            inverse 3D discrete Fourier transform of the input coefficients.
        """
        coeff = coeff[self.sort_pos]
        stacks = np.reshape(coeff, (self.acq_num_slices, self.first_stack_len))
        first_fft = np.asarray([self.FT.adj_op(stacks[slice]).T
                                for slice in np.arange(stacks.shape[0])])
        # TODO fix for higher N, this is not a usecase
        # interpolate_kspace = interp1d(self.z_samples[:, 0],
        #                               first_fft, kind='zero',
        #                               axis=0, bounds_error=False,
        #                               fill_value=0)
        # first_fft = interpolate_kspace(np.linspace(-0.5, 0.5,
        #                                            self.num_slices,
        #                                            endpoint=False))
        final = np.fft.ifftshift(np.fft.ifft(
            np.asarray(np.fft.fftshift(first_fft, axes=0)),
            axis=0, n=self.num_slices, norm="ortho"),
            axes=0)
        return final.T
