import torch
import torch.nn.functional as F
import numpy as np

from filters import convolve2d

## Implementation adapted from https://github.com/uschmidt83/fourier-deconvolution-network/blob/master/fdn_predict.py


def edgetaper_alpha(kernel, img_shape):
    z = torch.fft.fft(torch.sum(kernel, -1), img_shape[0]-1)
    za = torch.abs(z)
    z = torch.real(torch.fft.ifft(za * za))
    z = torch.cat([z, z[..., 0:1]], dim=-1)
    v1 = 1 - z / torch.max(z)

    z = torch.fft.fft(torch.sum(kernel, -2), img_shape[1] - 1)
    za = torch.abs(z)
    z = torch.real(torch.fft.ifft(za * za))
    z = torch.cat([z, z[..., 0:1]], dim=-1)
    v2 = 1 - z / torch.max(z)

    return v1.unsqueeze(-1) * v2.unsqueeze(-2)


def edgetaper(img, kernel, n_tapers=3, method='fft'):
    h, w = img.shape[-2:]
    alpha = edgetaper_alpha(kernel, (h, w))
    _kernel = kernel
    for i in range(n_tapers):
        blurred = convolve2d(img, kernel, method=method)
        img = alpha * img + (1.0 - alpha) * blurred
    return img

