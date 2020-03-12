from ..base import OperatorBase

from modopt.base.backend import get_array_module, \
    move_to_device, convert_to_tensor, convert_to_cupy_array
import pysap

import torch.nn as nn
import numpy as np
import torch
import itertools
import pywt
import warnings


class WaveNet(nn.Module):
    def __init__(self, wavelet, dim=2, padding='constant'):
        super(WaveNet, self).__init__()
        self.dimension = dim
        self.padding_mode = padding
        # Generate impulse of the vol_shape
        dec_filters = torch.stack(
            [torch.Tensor(wavelet.dec_hi[::-1]), torch.Tensor(wavelet.dec_lo[::-1])], dim=0)
        rec_filters = torch.stack(
            [torch.Tensor(wavelet.rec_hi), torch.Tensor(wavelet.rec_lo)], dim=0)
        dec_filters = torch.stack([
            torch.Tensor(torch.stack(tup))
            for tup in itertools.product(dec_filters, repeat=self.dimension)])
        rec_filters = torch.stack([
            torch.Tensor(torch.stack(tup))
            for tup in itertools.product(rec_filters, repeat=self.dimension)])
        self.wav_filters = []
        self.inv_filters = []
        for dec_f, rec_f in zip(dec_filters, rec_filters):
            wav_filter = dec_f[0]
            inv_filter = rec_f[0]
            for d in range(1, self.dimension):
                wav_filter = dec_f[d].unsqueeze(0) * wav_filter.unsqueeze(-1)
                inv_filter = rec_f[d].unsqueeze(0) * inv_filter.unsqueeze(-1)
            self.wav_filters.append(wav_filter)
            self.inv_filters.append(inv_filter)
        self.wav_filters = torch.stack(self.wav_filters[::-1])
        self.inv_filters = torch.stack(self.inv_filters[::-1])
        self.pad_max = len(wavelet.dec_hi) // 2 - 1
        # Setup the filter
        if self.dimension == 2:
            conv = nn.Conv2d
            inv_conv = nn.ConvTranspose2d
        else:
            conv = nn.Conv3d
            inv_conv = nn.ConvTranspose3d
        self.wavnet = conv(1, 2**self.dimension, wav_filter.shape, stride=2)
        self.inv_wavnet = inv_conv(2**self.dimension, 1, inv_filter.shape, stride=2)
        self.wavnet.weight.data = self.wav_filters[:, None]
        self.inv_wavnet.weight.data = self.inv_filters[:, None]

    def forward(self, vimg, levels):
        input_shape = list(vimg.shape[2:])
        mid_size = list(np.array(input_shape)//2)
        padded = nn.functional.pad(vimg, (self.pad_max,) * self.dimension * 2, mode=self.padding_mode)
        res = self.wavnet(padded)
        if levels > 1:
            res[:, :1] = self.forward(res[:, :1], levels-1)
        res = res.view(tuple([-1] + [2] * (self.dimension - 1) + mid_size))
        res = self.transpose(res)
        res = res.contiguous().view(tuple([-1, 1] + input_shape))
        return res

    def transpose(self, res, opp=False):
        X = np.arange(1, 2 * self.dimension - 2, 2)
        Y = np.arange(self.dimension, 2 * self.dimension - 1)
        if opp:
            X = X[::-1]
            Y = Y[::-1]
        for x, y in zip(X, Y):
            res = res.transpose(int(x), int(y))
        return res

    def adjoint(self, coeff, levels):
        input_shape = list(coeff.shape[2:])
        mid_size = list(np.array(input_shape)//2)
        twos = [2] * 2 * self.dimension
        twos[1::2] = mid_size
        image = coeff.view(twos)
        image = self.transpose(image, True)
        image = image.contiguous().view(tuple([-1] + [2**self.dimension] + mid_size)).clone()
        if levels>1:
            image[:, :1] = self.adjoint(image[:, :1], levels-1)
        image = self.inv_wavnet(image)
        if self.dimension == 3:
            image = image[:, :, self.pad_max:-self.pad_max,
                    self.pad_max:-self.pad_max, self.pad_max:-self.pad_max]
        else:
            image = image[:, :, self.pad_max:-self.pad_max, self.pad_max:-self.pad_max]
        return image


class WaveNetOp(OperatorBase):
    """ The 2D and 3D wavelet transform class, that uses Wavenet
    """

    def __init__(self, wavelet_name, nb_scale=4, dim=2, n_coils=1,
                 verbose=0, use_gpu=False, **kwargs):
        """ Initialize the 'WaveletN' class.

        Parameters
        ----------
        wavelet_name: str
            the wavelet name to be used during the decomposition.
        nb_scales: int, default 4
            the number of scales in the decomposition.
        n_coils: int, default 1
            the number of coils for multichannel reconstruction
        verbose: int, default 0
            the verbosity level.
        """
        self._nb_scale = nb_scale
        self.n_coils = n_coils
        self.verbose = verbose
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.wavenet = WaveNet(self.wavelet, dim, **kwargs)
        if use_gpu:
            try:
                self.wavenet = self.wavenet.cuda()
            except:
                warnings.warn('Could not launch on GPU!')
                use_gpu = False
        self.use_gpu = use_gpu

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
        # TODO this might have to be replaced in future when pysap.Image is removed
        if isinstance(data, pysap.Image):
            data = data.data
        xp = get_array_module(data)
        if xp == np and self.use_gpu is True:
            warnings.warn('Found data on CPU, moving to GPU.')
            data = move_to_device(data)
            xp = get_array_module(data)
        if self.n_coils == 1:
            data = data[xp.newaxis, :]
        data = data[xp.newaxis, :]
        coeff_real = convert_to_cupy_array(self.wavenet.forward(convert_to_tensor(data.real), self._nb_scale))
        coeff_imag = convert_to_cupy_array(self.wavenet.forward(convert_to_tensor(data.imag), self._nb_scale))
        coeff = coeff_real[:, 0] + 1j * coeff_imag[:, 0]
        if self.n_coils == 1:
            coeff = coeff[0]
        return coeff

    def adj_op(self, coeffs):
        """ Define the wavelet adjoint operator.
        This method returns the reconstructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        xp = get_array_module(coeffs)
        if xp == np and self.use_gpu is True:
            warnings.warn('Found data on CPU, moving to GPU.')
            coeffs = move_to_device(coeffs)
            xp = get_array_module(coeffs)
        if self.n_coils == 1:
            coeffs = coeffs[xp.newaxis, :]
        coeffs = coeffs[xp.newaxis, :]
        data_real = convert_to_cupy_array(self.wavenet.adjoint(convert_to_tensor(coeffs.real), self._nb_scale))
        data_imag = convert_to_cupy_array(self.wavenet.adjoint(convert_to_tensor(coeffs.imag), self._nb_scale))
        image = data_real[:, 0] + 1j * data_imag[:, 0]
        if self.n_coils == 1:
            image = image[0]
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
        fake_data[tuple(zip(shape // 2))] = 1

        data = self.op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)