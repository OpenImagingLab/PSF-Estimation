import math

import torch
import torch.nn.functional as F
import torch.fft
import numpy as np
from skimage import filters

import utils
from utils_fft import *


#####################################################################
######################## Convolution 2D #############################
#####################################################################

def convolve2d(img, kernel, ksize=25, padding='same', method='direct'):
    """
    A per kernel wrapper for torch.nn.functional.conv2d
    :param img: (B,C,H,W) torch.tensor, the input images
    :param kernel: (B,C,h,w) or 
                   (B,1,h,w) torch.tensor, the 2d blur kernels (valid for both deblurring methods), or 
                   [(B,C), (B,C), (B,C)] or 
                   [(B,1), (B,1), (B,1)], the separable 1d blur kernel parameters (valid only for spatial deblurring)
    :param padding: string, can be 'valid' or 'same' 
    : 
    :return imout: (B,C,H,W) torch.tensor, the filtered images
    """
    if method == 'direct':
        return conv2d_(img, kernel, padding)
    elif method == 'fft':
        X = torch.fft.fft2(utils.pad_with_kernel(img, kernel, mode='circular'))
        K = p2o(kernel, X.shape[-2:])
        return utils.crop_with_kernel( torch.real(torch.fft.ifft2(K * X)), kernel )
    else:
        raise('Convolution method %s is not implemented' % method)


def conv2d_(img, kernel, padding='same'):
    """
    Wrapper for F.conv2d with RGB images and kernels.
    """
    b, c, h, w = img.shape
    _, _, kh, kw = kernel.shape
    img = img.view(1, b*c, h, w)
    kernel = kernel.view(b*c, 1, kh, kw)
    return F.conv2d(img, kernel, groups=img.shape[1], padding=padding).view(b, c, h, w)


#####################################################################
####################### Bilateral filter ############################
#####################################################################


def extract_tiles(img, kernel_size, stride=1):
    b, c, _, _ = img.shape
    h, w = kernel_size
    tiles = F.unfold(img, kernel_size, stride)  # (B,C*H*W,L)
    tiles = tiles.permute(0, 2, 1)  # (B,L,C*H*W)
    tiles = tiles.view(b, -1, c, h ,w)
    return tiles


def bilateral_filter(I, ksize=5, sigma_spatial=1.5, sigma_color=0.1):
    ## precompute the spatial kernel: each entry of gw is a square spatial difference
    t = torch.arange(-ksize//2+1, ksize//2+1, device=I.device)
    xx, yy = torch.meshgrid(t, t, indexing='xy')
    gw = torch.exp(-(xx * xx + yy * yy) / (2 * sigma_spatial * sigma_spatial))  # (ksize, ksize)

    ## Create the padded array for computing the color shifts
    I_padded = utils.pad_with_kernel(I, ksize=ksize)

    ## Filtering
    var2_color = 2 * sigma_color * sigma_color
    return bilateral_filter_loop_(I, I_padded, gw, var2_color)


def bilateral_filter_loop_(I, I_padded, gw, var2, do_for=True):
    b, c, h, w = I.shape

    if do_for:  # memory-friendly option (Recommanded for larger images)
        J = torch.zeros_like(I)
        W = torch.zeros_like(I)
        for y in range(gw.shape[0]):
            yy = y + h
            # get the shifted image
            I_shifted = I_padded[..., y:yy, :]
            I_shifted = extract_tiles(I_shifted, kernel_size=(h,w), stride=1)  # (B,ksize,C,H,W)
            # color weight
            F = I_shifted - I.unsqueeze(1)  # (B,ksize,C,H,W)
            F = torch.exp(-F * F / var2) 
            # product with spatial weight
            F *= gw[y].view(-1, 1, 1, 1) # (B,ksize,C,H,W)
            J += torch.sum(F * I_shifted, dim=1)
            W += torch.sum(F, dim=1)
    else:  # pytorch-friendly option (Faster for smaller images and/or batch sizes)
        # get shifted images
        I_shifted = extract_tiles(I_padded, kernel_size=(h,w), stride=1)  # (B,ksize*ksize,C,H,W)
        F = I_shifted - I.unsqueeze(1)
        F = torch.exp( - F * F / var2)  # (B,ksize*ksize,C,H,W)
        # product with spatial weights
        F *= gw.view(-1, 1, 1, 1)
        J = torch.sum(F * I_shifted, dim=1)  # (B,C,H,W)
        W = torch.sum(F, dim=1)  # (B,C,H,W)
    del I_shifted  # manually free the memory occupied by I_shifted
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return J / (W + 1e-5)


#####################################################################
####################### Domain transform  ###########################
#####################################################################


def recursive_filter(I, sigma_s=60, sigma_r=0.4, num_iterations=3, joint_image=None):
    """
    (pytorch) Implementation of the edge aware smoothing with recursive filtering (EdgeAwareSmoothing Alg.6) from:
        [Eduardo Simoes Lopes Gastal and Manuel M. Oliveira. Domain transform for edge-aware
         image and video processing. ACM Transactions on Graphics (ToG), 30(4):69, 2011.]
    :param I: (B,C,H,W) torch.tensor, the input image(s)
    :param sigma_r: float, regularization parameter for domain transform 
    :param sigma_s: float, smoothness parameter for domain transform 
    :param num_iterations: int, iterations
    :param joint_image: (B,C,H,W) torch.tensor, the guide image(s) (optional)
    :return: img_smoothed: torch.tensor of same size as img, the smoothed image(s)
    """
    if joint_image is None:
        J = I
    else:
        J = joint_image

    batch, num_joint_channels, h, w = J.shape

    ## Compute the domain transform
    # Estimate horizontal and vertical partial derivatives using finite differences
    dIcdx = torch.diff(J, n=1, dim=-1)
    dIcdy = torch.diff(J, n=1, dim=-2)

    # compute the l1-norm distance of neighbor pixels.
    dIdx = torch.sum(torch.abs(dIcdx), dim=1)
    dIdx = torch.nn.functional.pad(dIdx.unsqueeze(0), pad=(1,0,0,0)).squeeze(0)
    dIdy = torch.sum(torch.abs(dIcdy), dim=1)
    dIdy = torch.nn.functional.pad(dIdy.unsqueeze(0), pad=(0,0,1,0)).squeeze(0)

    # compute the derivatives of the horizontal and vertical domain transforms
    dHdx = 1 + sigma_s/sigma_r * dIdx
    dVdy = 1 + sigma_s/sigma_r * dIdy

    # the vertical pass is performed using a transposed image
    dVdy = dVdy.transpose(-2, -1)

    ## Perform the filtering
    N = num_iterations
    F = I.clone()

    sigma_H = sigma_s
    for i in range(num_iterations):
        # Compute the sigma value for this iterations (Equation 14 of our paper)
        sigma_H_i = sigma_H * math.sqrt(3) * 2**(N - (i + 1)) / math.sqrt(4**N - 1)

        # Feedback coefficient (Appendix of our paper).
        a = math.exp(-math.sqrt(2) / sigma_H_i)
        
        V = (a**dHdx).unsqueeze(1)
        F = transformed_domain_recursive_filter_horizontal(F, V)
        F = F.transpose(-1, -2)

        V = (a**dVdy).unsqueeze(1)
        F = transformed_domain_recursive_filter_horizontal(F, V)
        F = F.transpose(-1, -2)

    return F


# @torch.jit.script
def transformed_domain_recursive_filter_horizontal(F, V):
    """
    (pytorch) Implementation of the recursive 1D (horizontal) filtering (Recursive1DFilter Alg.7) used in the edge aware smoothing from:
        [Eduardo Simoes Lopes Gastal and Manuel M. Oliveira. Domain transform for edge-aware
         image and video processing. ACM Transactions on Graphics (ToG), 30(4):69, 2011.]
    :param F: (B,C,H,W) torch.tensor, the input image(s)
    :param D: (B,1,H,W) torch.tensor, the filter used to control the diffusion
    :return: img_smoothed: torch.tensor of same size as img, the filtered image(s)
    """

    # Left -> Right filter
    for i in range(1, F.shape[-1], 1):
        F[..., i] += V[..., i] * (F[..., i - 1] - F[..., i])

    # Right -> Left filter
    for i in range(F.shape[-1]-2, -1, -1):  # from w-2 to 0
        F[..., i] += V[..., i + 1] * (F[..., i + 1] - F[..., i])

    return F



#####################################################################
###################### Classical filters ############################
#####################################################################


# @torch.jit.script
def fourier_gradients(images, freqs=None):
    """
    Compute the image gradients using Fourier interpolation as in Eq. (21a) and (21b)
        :param images: (B,C,H,W) torch.tensor
        :return grad_x, grad_y: tuple of 2 images of same dimensions as images that
                                are the vertical and horizontal gradients
    """
    ## Find fast size for FFT
    h, w = images.shape[-2:]
    h_fast, w_fast = images.shape[-2:]
    # h_fast = scipy.fft.next_fast_len(h)
    # w_fast = scipy.fft.next_fast_len(w)
    ## compute FT
    U = torch.fft.fft2(images)
    U = torch.fft.fftshift(U, dim=(-2, -1))
    ## Create the freqs components
    if freqs is None:
        freqh = (torch.arange(0, h_fast, device=images.device) - h_fast // 2).view(1,1,-1,1) / h_fast
        freqw = (torch.arange(0, w_fast, device=images.device) - w_fast // 2).view(1,1,1,-1) / w_fast
    else:
        freqh, freqw = freqs
    ## Compute gradients in Fourier domain
    gxU = 2 * np.pi * freqw * (-torch.imag(U) + 1j * torch.real(U))
    gxU = torch.fft.ifftshift(gxU, dim=(-2, -1))
    gxu = torch.real(torch.fft.ifft2(gxU))
    # gxu = crop(gxu, (h, w))
    gyU = 2 * np.pi * freqh * (-torch.imag(U) + 1j * torch.real(U))
    gyU = torch.fft.ifftshift(gyU, dim=(-2, -1))
    gyu = torch.real(torch.fft.ifft2(gyU))
    # gyu = crop(gyu, (h, w))
    return gxu, gyu


def crop(image, new_size):
    size = image.shape[-2:]
    if size[0] - new_size[0] > 0:
        image = image[..., :new_size[0], :]
    if size[1] - new_size[1] > 0:
        image = image[..., :new_size[1]]
    return image


def gaussian_filter(sigma, theta, shift=np.array([0.0, 0.0]), k_size=np.array([15, 15])):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1, lambda_2 = sigma
    theta = -theta

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1**2, lambda_2**2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position
    MU = k_size // 2 - shift
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calculate Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ))

    # Normalize the kernel and return
    if np.sum(raw_kernel) < 1e-2:
        kernel = np.zeros_like(raw_kernel)
        kernel[k_size[0]//2, k_size[1]//2] = 1
    else:
        kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


def dirac(dims):
    kernel = zeros(dims)
    hh = dims[0] // 2
    hw = dims[1] // 2
    kernel[hh, hw] = 1
    return kernel


def gaussian(images, sigma=1.0, theta=0.0):
    ## format Gaussian parameter for the gaussian_filter routine
    if isinstance(sigma, float) or isinstance(sigma, int):
        sigmas = ones(images.shape[0],2) * sigma
    elif isinstance(sigma, tuple) or isinstance(sigma, list):
        sigmas = ones(images.shape[0],2)
        sigmas[:,0] *= sigma[0]
        sigmas[:,1] *= sigma[1]
    else:
        sigmas = sigma
    if isinstance(theta, float) or isinstance(theta, int):
        thetas = ones(images.shape[0],1) * theta
    else:
        thetas = theta
    assert(theta.ndim-2)
    ## perform Gaussian filtering
    kernels = gaussian_filter(sigmas=sigmas, thetas=thetas)
    kernels = torch.to_tensor(kernels).unsqueeze(1).float().to(images.device)  # Nx1xHxW
    return conv2d(images, kernels)



def images_gradients(images, sigma=1.0):
    images_smoothed = fast_gaussian(images, sigma)
    gradients_x = torch.roll(images_smoothed, 1, dims=-2) - torch.roll(images_smoothed, -1, dims=-2)
    gradients_y = torch.roll(images_smoothed, 1, dims=-1) - torch.roll(images_smoothed, -1, dims=-1)
    return gradients_x, gradients_y



#####################################################################
####################### Fourier kernel ##############################
#####################################################################


### From here, taken from https://github.com/cszn/USRNet/blob/master/utils/utils_deblur.py
def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        # otf: NxCxHxWx2
        otf: NxCxHxW
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fft2(otf, dim=(-2, -1))
    return otf

def compute_gradient_magnitudes(gradients, rot, angles):
    gradient_x, gradient_y = gradients  # (B,C,H,W)
    gradient_x_gray = gradient_x.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
    gradient_y_gray = gradient_y.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
    angles = angles - rot - 90
    angles = (angles / 180 * np.pi).view(1, -1, 1, 1, 1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    gradient_magnitudes_angles = (cos * gradient_x_gray - sin * gradient_y_gray).abs()  # (B,N,1,H,W)
    gradient_magnitudes_angles = torch.amax(gradient_magnitudes_angles, dim=(-3, -2, -1))  # (B,N)
    return gradient_magnitudes_angles

def gradient_map(image):
    [h,w,c] = image.size()
    w_nums , h_nums = 10,10
    w_step, h_step  = w//w_nums , h//h_nums
    grads,grad = [],[]
    for x in range(w_nums):
        for y in range(h_nums):
            x_s, x_e = x * w_step, (x + 1) * w_step
            y_s, y_e = y * h_step, (y + 1) * h_step
            x_c, y_c = (x_s + x_e) // 2, (y_s + y_e) // 2
            y1 , x1 = torch.tensor(y_c - h // 2) ,torch.tensor(x_c - w // 2)
            rot = torch.rad2deg(torch.atan2(y1, x1))
            fov = int(math.sqrt((y_c - h // 2) ** 2 + (x_c - w // 2) ** 2))

            select = image[x_s:x_e, y_s:y_e]
            angles = torch.linspace(0, 180, 7).cuda()
            grad_patch = fourier_gradients(select, freqs=None)
            gradient_magnitudes_angles = compute_gradient_magnitudes(grad_patch, rot, angles)
            grads.append({'fov': fov, 'grads': gradient_magnitudes_angles})

    # fov = [item['fov'] for item in grads]
    data = [item['grads'] for item in grads]
    data1  = torch.stack(data,dim=1)

    return torch.sum(data1)

def compute_gradient(gradients):
    gradient_x, gradient_y = gradients  # (B,C,H,W)
    gradient_x_gray = gradient_x.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
    gradient_y_gray = gradient_y.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
    gradients = torch.concatenate([gradient_x_gray.squeeze(),gradient_y_gray.squeeze()],dim=-1)
    return gradients

def gradient(image,epsilon):
    [h,w,c] = image.size()
    w_nums , h_nums = 10,10
    w_step, h_step  = w//w_nums , h//h_nums
    grads,grad = [],[]
    for x in range(w_nums):
        for y in range(h_nums):
            x_s, x_e = x * w_step, (x + 1) * w_step
            y_s, y_e = y * h_step, (y + 1) * h_step
            x_c, y_c = (x_s + x_e) // 2, (y_s + y_e) // 2
            y1 , x1 = torch.tensor(y_c - h // 2) ,torch.tensor(x_c - w // 2)
            rot = torch.rad2deg(torch.atan2(y1, x1))
            fov = int(math.sqrt((y_c - h // 2) ** 2 + (x_c - w // 2) ** 2))

            select = image[x_s:x_e, y_s:y_e]
            grad_patch = fourier_gradients(select, freqs=None)
            gradients = compute_gradient(grad_patch)
            grads.append({'fov': fov, 'grads': gradients})

    # fov = [item['fov'] for item in grads]
    data = [item['grads'] for item in grads]
    data1  = torch.stack(data,dim=1)
    data1 = data1/torch.max(data1)
    # data2 = torch.where(abs(data1) > epsilon,1, data1**2/(epsilon**2))
    data2 = torch.where(abs(data1) > epsilon, 1, data1 ** 2)

    return torch.sum(data2)