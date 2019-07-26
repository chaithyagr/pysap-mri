# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains linears operators classes.
"""


# Package import
import pysap
from pysap.base.utils import flatten
from pysap.base.utils import unflatten

# Third party import
import numpy as np
from modopt.signal.wavelet import get_mr_filters, filter_convolve


class Wavelet2(object):
    """ The 2D wavelet transform class.
    """
    def __init__(self, wavelet_name, nb_scale=4, verbose=0, multichannel=False):
        """ Initialize the 'Wavelet2' class.
        Parameters
        ----------
        wavelet_name: str
            the wavelet name to be used during the decomposition.
        nb_scales: int, default 4
            the number of scales in the decomposition.
        verbose: int, default 0
            the verbosity level.
        """
        self.nb_scale = nb_scale
        self.multichannel = multichannel
        if wavelet_name not in pysap.AVAILABLE_TRANSFORMS:
            raise ValueError(
                "Unknown transformation '{0}'.".format(wavelet_name))
        transform_klass = pysap.load_transform(wavelet_name)
        self.transform = transform_klass(
            nb_scale=self.nb_scale, verbose=verbose)
        self.coeffs_shape = None
        self.flatten = flatten
        self.unflatten = unflatten

    def op(self, data):
        """ Define the wavelet operator.
        This method returns the input data convolved with the wavelet filter.
        Parameters
        ----------
        data: ndarray or Image
            input 2D data array.
        Returns
        -------
        coeffs: ndarray
            the wavelet coefficients.
        """
        if isinstance(data, np.ndarray):
            data = pysap.Image(data=data)
        if self.multichannel:
            coeffs = []
            self.coeffs_shape = []
            for channel in range(data.shape[0]):
                self.transform.data = data[channel]
                self.transform.analysis()
                coeff, coeffs_shape = self.flatten(self.transform.analysis_data)
                coeffs.append(coeff)
                self.coeffs_shape.append(coeffs_shape)
            return np.asarray(coeffs)
        else:
            self.transform.data = data
            self.transform.analysis()
            coeffs, self.coeffs_shape = self.flatten(self.transform.analysis_data)
            return coeffs

    def adj_op(self, coeffs, dtype="array"):
        """ Define the wavelet adjoint operator.
        This method returns the reconsructed image.
        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.
        dtype: str, default 'array'
            if 'array' return the data as a ndarray, otherwise return a
            pysap.Image.
        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        if self.multichannel:
            images = []
            for channel, coeffs_shape in zip(range(coeffs.shape[0]),
                                             self.coeffs_shape):
                self.transform.analysis_data = self.unflatten(coeffs[channel],
                                                              coeffs_shape)
                images.append(self.transform.synthesis().data)
            return np.asarray(images)
        else:
            self.transform.analysis_data = self.unflatten(coeffs, self.coeffs_shape)
            image = self.transform.synthesis()
            if dtype == "array":
                return image.data
            return image

    def l2norm(self, shape):
        """ Compute the L2 norm.
        Parameters
        ----------
        shape: uplet
            the data shape.
        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data
        shape = np.asarray(shape)
        shape += shape % 2
        fake_data = np.zeros(shape)
        fake_data[list(zip(shape // 2))] = 1

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)


class WaveletUD2(object):
    """The wavelet undecimated operator using pysap wrapper.
    """

    def __init__(self, wavelet_id, nb_scale=4, verbose=0, multichannel=False):
        """Init function for Undecimated wavelet transform

        Parameters
        -----------
        wavelet_id: int
            ID of wavelet being used
        nb_scale: int, default 4
            the number of scales in the decomposition.

        Private Variables:
            _has_run: Checks if the get_mr_filters was called already
        """
        self.wavelet_id = wavelet_id
        self.multichannel = multichannel
        self.nb_scale = nb_scale
        self._opt = [
            '-t{}'.format(self.wavelet_id),
            '-n{}'.format(self.nb_scale),
        ]
        self._has_run = False
        self._shape = (None,)

    def _get_filters(self, shape):
        """ Function to get the Wavelet coefficients of Delta[0][0]. This function is called only once and later the
        wavelet coefficients are obtained by convolving these coefficients with input Data
        """
        self.transform = get_mr_filters(
            tuple(shape),
            opt=self._opt,
            coarse=True,
        )
        self._has_run = True
        self._shape = shape

    def op(self, data):
        """ Define the wavelet operator.

        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data: ndarray or Image
            input 2D data array.

        Returns
        -------
        coeffs: ndarray
            the wavelet coefficients.
        """
        if not self._has_run or data.shape != self._shape:
            if self.multichannel:
                self._get_filters(list(data.shape)[1:])
            else:
                self._get_filters(data.shape)
        if self.multichannel:
            coefs_real = np.array([filter_convolve(data[channel].real, self.transform)
                                   for channel in np.arange(data.shape[0])])
            coefs_imag = np.array([filter_convolve(data[channel].real, self.transform)
                                   for channel in np.arange(data.shape[0])])
        else:
            coefs_real = filter_convolve(data.real, self.transform)
            coefs_imag = filter_convolve(data.imag, self.transform)
        return coefs_real + 1j * coefs_imag

    def adj_op(self, coefs):
        """ Define the wavelet adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.
        dtype: str, default 'array'
            if 'array' return the data as a ndarray, otherwise return a
            pysap.Image.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        if not self._has_run:
            raise RuntimeError(
                "`op` must be run before `adj_op` to get the data shape",
            )
        if self.multichannel:
            data_real = np.array([filter_convolve(coefs.real[channel], self.transform, filter_rot=True)
                                  for channel in np.arange(coefs.imag.shape[0])])
            data_imag = np.array([filter_convolve(coefs.imag[channel], self.transform, filter_rot=True)
                                  for channel in np.arange(coefs.imag.shape[0])])
        else:
            data_real = filter_convolve(coefs.real, self.transform, filter_rot=True)
            data_imag = filter_convolve(coefs.imag, self.transform, filter_rot=True)
        return data_real + 1j * data_imag

    def l2norm(self, shape):
        """ Compute the L2 norm.
        Parameters
        ----------
        shape: uplet
            the data shape.
        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data
        shape = np.asarray(shape)
        shape += shape % 2
        fake_data = np.zeros(shape)
        fake_data[tuple(zip(shape // 2))] = 1

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)

