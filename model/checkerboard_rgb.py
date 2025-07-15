from itertools import chain
from math import ceil
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
from math import *
from skimage.feature import corner_harris, corner_subpix, corner_peaks
import torch
import math
import torch.nn as nn
from torch.fft import fftshift, fft2,ifft2
import torchvision
import torch.nn.functional as F
from scipy.ndimage import shift
# import tensorflow as tf
from scipy.signal import convolve2d
import pandas as pd
import os
import utils
import utils_fft
import filters
import edgetaper
import argparse
from scipy.io import savemat
# import images


class checker():

    def __init__(self, square_size=None, device=torch.device('cpu')):

        super(checker, self).__init__()
        self.device = device
        # Move all variables to device.
        self.to(device)
        self.square_size = square_size

    def to(self, device=torch.device('cpu')):
        """ Move all variables to target device.
        """
        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec('self.{x} = self.{x}.to(device)'.format(x=key))
            elif val.__class__.__name__ in ('list', 'tuple'):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        exec('self.{x}[{i}] = self.{x}[{i}].to(device)'.format(x=key, i=i))

        self.device = device
        return self

    def latent(self, IS ):
        """ generate checkerboard according to the HFOV.
            suppose H:W = 3:4
        """
        rows, cols = int(2*IS.diag), int(2*IS.diag)
        square_size = self.square_size
        erows, ecols = rows+2000, cols+2000

        # Create a blank white image
        # image = np.ones((erows, ecols), dtype=np.uint8) * 255
        image = np.ones((erows, ecols), dtype=np.uint8) * 220

        # Generate the checkerboard pattern
        for row in range(erows):
            for col in range(ecols):
                if (row // square_size + col // square_size) % 2 == 0:
                    image[row, col] = 15
                    # image[row, col] = 0
        # Rotate the image by the specified skew angle
        center = (erows // 2, ecols // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -5, 1)
        rot_img = cv2.warpAffine(image, rotation_matrix, (ecols, erows))
        rot_img = rot_img.squeeze()
        # Leave a side with a width of 20
        cropped_image = rot_img[center[0]-rows//2-200:center[0]+rows//2+200, center[1]-cols//2-200:center[1]+cols//2+200]
        self.latent = cropped_image


    @staticmethod
    def rotate(psf, angle):
        if not isinstance(psf, torch.Tensor):
            psf = torch.tensor(psf)
        if len(psf.shape) == 2:
            psf = psf.unsqueeze(0).unsqueeze(0)
        elif len(psf.shape) == 3:
            psf = psf.permute(2,0,1).unsqueeze(0)
        rot = torch.deg2rad(torch.tensor(angle))
        theta = torch.tensor([[torch.cos(rot), -torch.sin(rot), 0], [torch.sin(rot), torch.cos(rot), 0]]).view(1, 2, 3)
        grid = F.affine_grid(theta, size=psf.size())
        rotated_psf = F.grid_sample(psf.float(),grid.float())
        rotated_psf = rotated_psf.squeeze()
        return rotated_psf

    ####### crop checkboard to edges #######
    def crop(self,IS,savepath):
        image = self.latent
        coords = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.05)
        h ,w = IS.res[0], IS.res[0]
        pixelsize , EFL = IS.pixelsize, IS.efl
        h_coords, v_coords = coords, coords
        v_coords  = v_coords + [self.square_size//2,-self.square_size//12]
        ewid = 2* IS.s_psf
        center = IS.res[0]//2
        shot_noise, read_noise = 0, 0
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        # path = os.path.join(savepath, 'pairs')
        # if not os.path.exists(path):
        #     os.makedirs(path)
        psfs = IS.Zer2PSF2(IS, Num=IS.hfov * 10 + 1)

        for i in range(coords.shape[0]):
            cropped_v = image[v_coords[i,0]-ewid:v_coords[i,0] + ewid, v_coords[i,1] - ewid:v_coords[i,1] + ewid]
            vy, vx = (h // 2 - v_coords[i, 0]), (v_coords[i, 1] - w // 2)
            fov = np.rad2deg(np.arctan(np.sqrt(vx ** 2 + vy ** 2) * pixelsize / EFL))
            if cropped_v.shape[0]*cropped_v.shape[1] == (2*ewid)**2 and fov <= IS.hfov:
                rot = np.degrees(np.arctan2(vy, vx)) - 90- 5.0
                rot = - rot
                rot = rot.astype(float)
                fov = np.rad2deg(np.arctan(np.sqrt(vx ** 2 + vy ** 2) * pixelsize / EFL))
                if abs(fov) <= abs(IS.hfov):
                    blur_crop_l , blur ,psf = [],[],[]
                    nor_img = cropped_v/255
                    nor_img = torch.from_numpy(nor_img.astype(np.float32))
                    nor_img = nor_img.unsqueeze(-1).repeat(1,1,3)
                    blur_crop = checker.convolve1(nor_img, psfs, IS, fov, rot, IS.s_psf - 1)
                    ## add noise
                    noise = torch.normal(mean=0, std=0, size= blur_crop.size())
                    blur_crop = torch.clip(blur_crop+noise,0,1)
                    blur_img = blur_crop.mul(255).to(torch.uint8).numpy()
                    blur_img = cv2.cvtColor(blur_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(savepath , 'x%d_y%d_fov%.2f_angle%.2f_v.tif' % (vx, vy, fov, rot)),blur_img)


    @staticmethod
    def convolve1(image,psfs,IS,fov,rot,wid):
        """ generate blurry checkerboard by path convolve with the psfs. """
        idx = int(10*fov)
        # print(fov)
        psf1  = psfs[idx]
        psf1 = checker.downsample(IS,psf1)
        psf = checker.rotate(psf1, rot)
        image = image.permute(2,0,1).unsqueeze(0)
        psf = psf.unsqueeze(1)
        convolved_image = F.conv2d(image, psf, groups=3, padding='valid', bias=None, stride=1)
        convolved_image = convolved_image.squeeze()
        c,h,w = convolved_image.shape
        return convolved_image[:,h//2-wid:h//2+wid,w//2-wid:w//2+wid].permute(1,2,0)

    @staticmethod
    def rotate(psf, angle):
        if not isinstance(psf, torch.Tensor):
            psf = torch.tensor(psf)
        if len(psf.shape) == 2:
            psf = psf.unsqueeze(0).unsqueeze(0)
        elif len(psf.shape) == 3:
            psf = psf.permute(2, 0, 1).unsqueeze(0)
        rot = torch.deg2rad(torch.tensor(angle))
        theta = torch.tensor([[torch.cos(rot), -torch.sin(rot), 0], [torch.sin(rot), torch.cos(rot), 0]]).view(1, 2, 3)
        grid = F.affine_grid(theta, size=psf.size())
        rotated_psf = F.grid_sample(psf.float(), grid.float())
        rotated_psf = rotated_psf.squeeze()
        return rotated_psf

    @staticmethod
    def downsample(IS,psf):
        """ PSF size only influence the resolution of MTF/SFR, not influence the absolute value, such as MTF50"""
        if not isinstance(psf, torch.Tensor):
            psf = torch.tensor(psf)
        if len(psf.shape)==2:
            psf = psf.unsqueeze(0).unsqueeze(0)
            c= 1
        elif len(psf.shape)==3:
            psf = psf.permute(2,0,1).unsqueeze(0)
            c = 3
        size = (IS.s_psf, IS.s_psf)
        psf = F.interpolate(psf, size=size, mode='bilinear', antialias=True)
        psf = psf.squeeze()
        for i in range(c):
            psf[i,...] = psf[i,...]  / torch.sum(psf[i,...])
        return psf.permute(1,2,0)



    @staticmethod
    def random_noise_levels_log(shot_noise=None):
        """Generates random noise levels from a log-log linear distribution."""
        if shot_noise is None:
            log_min_shot_noise = np.log(0.0001)
            log_max_shot_noise = np.log(0.012)
            log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
            shot_noise = np.exp(log_shot_noise)
        else:
            log_shot_noise = np.log(shot_noise)
        line = lambda x: 2.18 * x + 1.20
        log_read_noise = line(log_shot_noise) + np.random.normal(0, 0.26)
        read_noise = np.exp(log_read_noise)
        return shot_noise, read_noise

    @staticmethod
    def add_noise(image, shot_noise, read_noise):
        """Adds random shot (proportional to image) and read (independent) noise."""
        variance = image * shot_noise + read_noise
        # noise = []
        # for i in range(20):
        #     noise.append(np.random.normal(0, np.sqrt(variance), size=variance.shape))
        # noise = np.stack(noise, axis=-1)
        # noise = np.mean(noise,axis=-1)
        noise = np.random.normal(0, np.sqrt(variance), size=variance.shape)
        return image + noise


    @staticmethod
    def mosaic(image):
        """Extracts RGGB Bayer planes from an RGB image."""
        # image = image.permute(1, 2, 0)  # Permute the image tensor to HxWxC format from CxHxW format
        shape = image.size()
        red = image[0::2, 0::2, 0]
        green_red = image[0::2, 1::2, 1]
        green_blue = image[1::2, 0::2, 1]
        blue = image[1::2, 1::2, 2]
        out = torch.stack((red, green_red, green_blue, blue), dim=-1)
        # out = torch.stack((red, green_blue, blue), dim=-1)
        out = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
        # out = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 3))
        out = out.permute(2, 0, 1)  # Re-Permute the tensor back to CxHxW format
        return out

    @staticmethod
    def demosaic(bayer_images):
        def SpaceToDepth_fact2(x):
            # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
            bs = 2
            N, C, H, W = x.size()
            x = x.view(N, C, H // bs, bs, W // bs, bs)  # (N, C, H//bs, bs, W//bs, bs)
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
            x = x.view(N, C * (bs ** 2), H // bs, W // bs)  # (N, C*bs^2, H//bs, W//bs)
            return x

        def DepthToSpace_fact2(x):
            # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
            bs = 2
            N, C, H, W = x.size()
            x = x.view(N, bs, bs, C // (bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
            x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
            x = x.view(N, C // (bs ** 2), H * bs, W * bs)  # (N, C//bs^2, H * bs, W * bs)
            return x

        """Bilinearly demosaics a batch of RGGB Bayer images."""
        if len(bayer_images.shape) == 3:
            bayer_images = bayer_images.unsqueeze(0)
        bayer_images = bayer_images.permute(0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format

        shape = bayer_images.size()
        shape = [shape[1] * 2, shape[2] * 2]

        red = bayer_images[Ellipsis, 0:1]
        upsamplebyX = nn.Upsample(size=shape, mode='bilinear', align_corners=False)
        red = upsamplebyX(red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        green_red = bayer_images[Ellipsis, 1:2]
        green_red = torch.flip(green_red, dims=[1])  # Flip left-right
        green_red = upsamplebyX(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        green_red = torch.flip(green_red, dims=[1])  # Flip left-right
        green_red = SpaceToDepth_fact2(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        green_blue = bayer_images[Ellipsis, 2:3]
        green_blue = torch.flip(green_blue, dims=[0])  # Flip up-down
        green_blue = upsamplebyX(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        green_blue = torch.flip(green_blue, dims=[0])  # Flip up-down
        green_blue = SpaceToDepth_fact2(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        green_at_red = (green_red[Ellipsis, 0] + green_blue[Ellipsis, 0]) / 2
        green_at_green_red = green_red[Ellipsis, 1]
        green_at_green_blue = green_blue[Ellipsis, 2]
        green_at_blue = (green_red[Ellipsis, 3] + green_blue[Ellipsis, 3]) / 2

        green_planes = [
            green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
        ]
        green = DepthToSpace_fact2(torch.stack(green_planes, dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        blue = bayer_images[Ellipsis, 3:4]
        blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])
        blue = upsamplebyX(blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])

        rgb_images = torch.cat([red, green, blue], dim=-1)
        rgb_images = rgb_images.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
        return rgb_images.squeeze().permute(1,2,0)
