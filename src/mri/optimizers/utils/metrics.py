# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from modopt.math.metrics import ssim, psnr
import numpy as np


def box_ssim(y_true, y_pred, mean_factor=0.7):
    return ssim(y_true, y_pred, y_true > mean_factor * np.mean(y_true))


def box_psnr(y_true, y_pred, mean_factor=0.7):
    return psnr(y_true, y_pred, y_true > mean_factor * np.mean(y_true))