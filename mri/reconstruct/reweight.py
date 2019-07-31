# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Reweighting optimisation strategies.
"""


# Package import
from pysap.base.utils import flatten

# Third party import
import numpy as np
from modopt.math.stats import sigma_mad


class mReweight(object):
    """ Ming reweighting.

    This class implements the reweighting scheme described in Ming2017.

    Parameters
    ----------
    weights: ndarray
        Array of weights
    linear_op: pysap.numeric.linear.Wavelet
        A linear operator.
    thresh_factor: float, default 1
        Threshold factor: sigma threshold.
    """
    def __init__(self, weights, linear_op, thresh_factor=1):
        self.weights = weights
        self.original_weights = np.copy(self.weights)
        self.thresh_factor = thresh_factor
        self.linear_op = linear_op

    def reweight(self, x_new):
        """ Update the weights.
            Implements a multichannel version, however the reweighting is done by measuring
            the noise on per band basis than per scale like earlier.
            TODO need to fix this, but it could be a good approximation and now we will have weights per band

        Parameters
        ----------
        x_new: ndarray
            the current primal solution.

        Returns
        -------
        sigma_est: ndarray
            the variance estimate on each scale.
        """
        coeffs = self.linear_op.op(x_new)
        if self.linear_op.multichannel:
            all_channel_weights = []
            for channel, coeffs_shape in zip(range(coeffs.shape[0]),
                                             self.linear_op.coeffs_shape):
                coeff_ch = self.linear_op.unflatten(coeffs[channel],
                                                              coeffs_shape)
                sigma_est = []
                weights = []
                for coeff, band in zip(coeff_ch, coeffs_shape):
                    coeffs_in_band_array, _ = flatten(coeff)
                    std_at_band_i = sigma_mad(coeffs_in_band_array)
                    sigma_est.append(std_at_band_i)
                    thr = np.ones(coeffs_in_band_array.shape, dtype=self.weights.dtype)
                    thr *= self.thresh_factor * std_at_band_i
                    weights.append(thr)
                flattened_weights, _ = self.linear_op.flatten(weights)
                all_channel_weights.append(flattened_weights)
            self.weights = np.asarray(all_channel_weights)
        else:
            coeff_unflattened = self.linear_op.unflatten(coeffs, self.linear_op.coeffs_shape)
            sigma_est = []
            weights = []
            for coeff, band in zip(coeff_unflattened, self.linear_op.coeffs_shape):
                coeffs_in_band_array, _ = flatten(coeff)
                std_at_band_i = sigma_mad(coeffs_in_band_array)
                sigma_est.append(std_at_band_i)
                thr = np.ones(coeffs_in_band_array.shape, dtype=self.weights.dtype)
                thr *= self.thresh_factor * std_at_band_i
                weights.append(thr)
            self.weights, _ = self.linear_op.flatten(weights)
        return sigma_est
