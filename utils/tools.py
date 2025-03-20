import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.fft import fftshift, fft2
from torchvision import datasets, transforms
import torch.nn.functional as F
import glob
import numpy as np
import torch
import random
# import h5py
import matplotlib.pyplot as plt
import os
import yaml
import cv2
import math
import time
import torch.utils.data as data
# from . import unprocess_torch
# from . import process_torch
# from unprocess_torch import *
# import process_torch


# class Dataset_from_h5(data.Dataset):
#
#     def __init__(self, src_path, train=True):
#         self.path = src_path
#         self.if_train = train
#         h5f = h5py.File(self.path, 'r')
#         self.keys = list(h5f.keys())
#         if self.if_train:
#             random.shuffle(self.keys)
#         h5f.close()
#
#     def __getitem__(self, index):
#         h5f = h5py.File(self.path, 'r')
#         key = self.keys[index]
#         data = np.array(h5f[key]).reshape(h5f[key].shape)
#         h5f.close()
#         sfr,weights,rot,fov = np.ascontiguousarray(data[0,:]),data[1,0],data[2,0],data[3,0]
#         return sfr,weights,rot,fov
#
#     def __len__(self):
#         return len(self.keys)

def gcd(a, b):
    # 辗转相除法计算最大公约数
    while b != 0:
        a, b = b, a % b
    return a

def lcm(a, b):
    # 计算最小公倍数
    return (a * b) // gcd(a, b)

# 计算最小公倍数，考虑0值
def tensor_lcm(tensor):
    # 初始化结果为第一个非零元素的绝对值
    result = None
    for element in tensor:
        if element != 0:
            result = abs(element)
            break
    if result is None:
        # 如果所有元素都为0，返回0
        return 0
    for element in tensor:
        if element != 0:
            result = lcm(result, abs(element))
    return result

def weights(angles,bins=4):
    # 根据输入的角度值确定每个的权重
    A = torch.histc(angles, bins=bins,min=0,max=0)
    A = A.int()
    lcm_result = tensor_lcm(A)
    # 统计每个角度的出现次数
    tensor = torch.where(A == 0, float('inf'), A)
    weight = lcm_result/tensor
    C = [b for a, b in zip(A, weight) for _ in range(a)]
    C = torch.tensor(C)
    return C/sum(C)

def weights2(nums):
    # 根据输入的出现次数确定每个的权重
    A = nums
    lcm_result = tensor_lcm(A)
    # 统计每个角度的出现次数
    tensor = torch.where(A == 0, float('inf'), A)
    weight = lcm_result/tensor
    return weight/sum(weight)

def slice(mtf_2d, rot):
    device = rot.device
    batch_size = mtf_2d.shape[0]
    size = mtf_2d.shape[-1]//(2*5) + 1
    "center should shift for odd number in interpolate in MTF calculation"
    center = (mtf_2d.shape[-1]) // 2
    if not torch.is_tensor(rot):
        rot = torch.tensor(rot)

    x = torch.linspace(0, size-1, size)  # 生成 x 坐标序列
    y = torch.zeros_like(x)  # 生成对应的 y 坐标序列
    coordinates = torch.stack((x, y), dim=1).to(device)
    rot_r = torch.deg2rad(rot)
    cos_rot = torch.cos(rot_r).unsqueeze(-1)
    sin_rot = torch.sin(rot_r).unsqueeze(-1)
    rot_matrix = torch.cat((cos_rot, -sin_rot, sin_rot, cos_rot), dim=-1).view(-1, 2, 2).float().to(device) # 旋转矩阵

    rot_coord = torch.matmul(coordinates.unsqueeze(0), rot_matrix).squeeze(0) * 5
    rot_coord = rot_coord.round().to(torch.int) + torch.tensor((center, center)).to(device)

    # selected_data = mtf_2d[torch.arange(batch_size).unsqueeze(1), rot_coord[..., 0], rot_coord[..., 1]]

    r = 2*torch.tensor(center) - rot_coord[...,1]
    c = rot_coord[...,0]

    selected_data = mtf_2d[torch.arange(batch_size).unsqueeze(1), r, c]
    return selected_data


# def PSF2MTF_B(psf, size):
#     if isinstance(psf, np.ndarray):
#         psf = torch.from_numpy(psf)
#     if len(psf.shape) ==2:
#         psf = psf.unsqueeze(0)
#     N = psf.shape[1]
#     psf = psf.unsqueeze(1)
#     left, top = (size - N) // 2, (size - N) // 2
#     right, bottom = size - N - left, size - N - top
#     psf = F.pad(psf, (left, right, top, bottom), value=0)
#     """ SFR interpolate factor =2"""
#     # psf = F.interpolate(psf, scale_factor=2.0, mode='bilinear', align_corners=False)
#     psf = torch.squeeze(psf)
#     mtf_batch = []
#     if len(psf.shape)==2:
#         psf = psf.unsqueeze(0)
#     batch_size = psf.shape[0]
#     for i in range(batch_size):
#         # complex = fftshift(fft2(psf[i]))
#         complex = fft2(psf[i])
#         complex = fftshift(complex)
#         mtf = abs(complex)
#         mtf = mtf / torch.max(torch.max(mtf))
#         # H = transforms.CenterCrop(size)
#         # mtf = H(mtf)
#         mtf_batch.append(mtf)

#     mtf_batch = torch.stack(mtf_batch)

#     return mtf_batch

def PSF2MTF(psf, size):
    if isinstance(psf, np.ndarray):
        psf = torch.from_numpy(psf)
    if len(psf.shape) ==2:
        psf = psf.unsqueeze(0)
    N = psf.shape[1]
    psf = psf.unsqueeze(1)
    left, top = (size - N) // 2, (size - N) // 2
    right, bottom = size - N - left, size - N - top
    psf = F.pad(psf, (left, right, top, bottom), value=0)
    """ SFR interpolate factor =2"""
    # psf = F.interpolate(psf, scale_factor=2.0, mode='bilinear', align_corners=False)
    psf = torch.squeeze(psf)
    mtf_batch = []
    if len(psf.shape)==2:
        psf = psf.unsqueeze(0)
    batch_size = psf.shape[0]
    for i in range(batch_size):
        # complex = fftshift(fft2(psf[i]))
        complex = fft2(psf[i])
        complex = fftshift(complex)
        mtf = abs(complex)
        mtf = mtf / torch.max(torch.max(mtf))
        # H = transforms.CenterCrop(size)
        # mtf = H(mtf)
        mtf_batch.append(mtf)

    mtf_batch = torch.stack(mtf_batch)

    return mtf_batch

def downsample(psf, size):
    if isinstance(psf, np.ndarray):
        psf = torch.from_numpy(psf)
    if len(psf.shape) == 3:
        psf = psf.unsqueeze(1)
    elif  len(psf.shape) == 2:
        psf = psf.unsqueeze(0).unsqueeze(0)
    kernel = F.interpolate(psf, size = size, mode='bilinear', align_corners=False)
    kernel = kernel.squeeze() / torch.sum(kernel)
    return kernel.squeeze()


def downsample_B(psf, scale):
    if isinstance(psf, np.ndarray):
        psf = torch.from_numpy(psf)
    if len(psf.shape) == 3:
        if psf.shape[0]>psf.shape[2]:
            psf = psf.permute(2,0,1)
        psf = psf.unsqueeze(0)
    elif  len(psf.shape) == 2:
        psf = psf.unsqueeze(0).unsqueeze(0)
    kernel = F.interpolate(psf, scale_factor=scale, mode='bilinear', align_corners=False)
    if kernel.shape[1]< kernel.shape[-1]:
        kernel = kernel.permute(0,2,3,1)
    return kernel.squeeze()

def multi_imshow(psf,filename =None,title = None):
    w = len(psf)
    if title is not None:
        tl = len(title)
    if w < 6:
        fig = plt.figure(figsize=(int(5.1*w), 5), dpi=100)
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
        for i in range(w):
           figname = fig.add_subplot(1, w, i+1)
           figname.imshow(psf[i],cmap='bone')
           if  title is not None and i < tl:
              figname.set_title(title[i])
    elif w <=20:
        fig = plt.figure(figsize=(w//2*5, int(10.2)), dpi=100)
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
        for i in range(w):
           figname = fig.add_subplot(2, (w+1)//2, i+1)
           figname.imshow(psf[i],cmap='bone')
           if  title is not None and i < tl:
              figname.set_title(title[i])
    else:
        fig = plt.figure(figsize=(w//3*5, int(15.2)), dpi=100)
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
        for i in range(w):
           figname = fig.add_subplot(3, (w+1)//3, i+1)
           figname.imshow(psf[i],cmap='bone')
           if  title is not None and i < tl:
              figname.set_title(title[i])
    if filename is None:
        plt.show()
    else:
        savefilepath = os.path.join('.\image', filename + '.png')
        plt.savefig(savefilepath, bbox_inches='tight',overwrite='True')

def cal_psnr(original, reconstructed):
    # 将图像转换为浮点数
    if isinstance(original,np.ndarray):
        original = torch.from_numpy(original).cuda()
    if isinstance(reconstructed,np.ndarray):
        reconstructed = torch.from_numpy(reconstructed).cuda()
    original = original.squeeze()
    reconstructed = reconstructed.squeeze()
    # 计算 MSE（均方误差）
    mse = torch.mean((original.cuda().float() - reconstructed.cuda().float()) ** 2)
    # 计算 PSNR
    max_pixel_value = torch.max(original)
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))

    return psnr

def rotatepsf(psf,angle):
    psf=psf.unsqueeze(0).unsqueeze(0).cpu()
    rot= torch.tensor(angle)
    theta= torch.tensor([[torch.cos(rot), -torch.sin(rot), 0],[torch.sin(rot), torch.cos(rot), 0]]).view(1, 2, 3)
    grid = F.affine_grid(theta, size=psf.size())
    rotated_psf= F.grid_sample(psf.float(), grid.float())#psf1.view(1, 1, psf1.size(0), psf1.size(1)).double()double())
    rotated_psf=rotated_psf.squeeze()
    return rotated_psf
    # return rotated_psf/torch.sum(rotated_psf)

def fov2H(fov,IS):
    'input: fov field of view (unit:degree),output: relative normalized field height'
    # H = math.tan(math.radians(fov)) * IS.efl / IS.pixelsize/ IS.diag
    device = IS.device
    H = torch.sin(torch.deg2rad(fov.squeeze())) / torch.sin(torch.deg2rad(torch.tensor(IS.hfov)))
    return H.to(device)

def H2fov(H,IS):
    'input: relative normalized field height,output:fov field of view (unit:degree)'
    fov = math.degrees(math.atan(H * IS.pixelsize * IS.diag / IS.efl))
    return int(10*fov)/10

def psf_map(psfs,IS):
    'input: psf list, output: psf_map'
    # M = int(np.sqrt(2)*len(psfs))
    # X ,Y = int(IS.res[0]/np.sqrt(IS.res[0]**2 + IS.res[1]**2) * 2 *len(psfs)) , int(IS.res[1]/np.sqrt(IS.res[0]**2 + IS.res[1]**2) * 2 *len(psfs))
    X, Y = IS.psf_line
    num = len(psfs)
    Hs = torch.linspace(0,1,num)
    channel= psfs[0].size(2)
    if len(psfs[0].size()) == 3:
        [h,w,c] = psfs[0].size()
    else:
        [h,w] = psfs[0].size()
        c = 1
    psf_list = []
    for i in range(X):
        for j in range(Y):
            cy , cx = ((X-1)/2-i) , (j-(Y-1)/2)
            H = np.sqrt(cx**2 +cy**2)/np.sqrt(((X-1)/2)**2 +((Y-1)/2)**2)
            index = torch.argmin(torch.abs(H-Hs))
            # print(index)
            rot =  np.arctan2(cy,cx)  - np.pi/2
            psf= []
            for color in range(channel):
                psf_single = psfs[index][:, :, color]
                rot_psf = rotatepsf(psf_single, rot)
                psf.append(rot_psf)
            psf = torch.stack(psf,dim=-1)
            psf_list.append(psf)
    psf_map = torch.stack(psf_list).view(X, Y, h, w,c)
    psfmap = psf_map.permute(0, 2, 1, 3,4).contiguous().view(X*h, Y*w,c)
    return psf_map,psfmap


def blurry(latent,IS,psf_list):
    """ generate blurry checkerboard by path convolve with the psfs. """
    erows, ecols,channel = latent.shape
    crop_unit = 300
    delta = crop_unit // 2
    centerx, centery = ecols//2, erows//2
    X = np.arange(delta, ecols, crop_unit)
    Y = np.arange(delta, erows, crop_unit)
    XX, YY = np.meshgrid(X, Y)
    nums  = XX.shape[0] * XX.shape[1]
    pix ,EFL = IS.pixelsize , IS.efl
    blurry = np.zeros_like(latent)
    for color in range(channel):
        psfs = psf_list[color]
        for i in range(nums):
            rows, cols = i//XX.shape[1], i%XX.shape[1]
            x, y = XX[rows,cols], YY[rows,cols]
            x1 = x - centerx
            y1 = centery - y
            fov = np.degrees(np.arctan(np.sqrt(x1 ** 2 + y1 ** 2) * pix / EFL))
            fov = round(10 * fov) / 10
            rot = np.degrees(np.arctan2(y1, x1))
            psf1 = [item['psf'] for item in psfs if item['fov'] == fov][0]
            psf = rotatepsf(psf1, np.deg2rad(rot - 90))
            psf = psf.numpy()
            psf = psf/np.sum(psf)
            # psf = downsample(psf, IS.s_psf)
            convolved_image = cv2.filter2D(latent[:,:,color], -1, psf, borderType = cv2.BORDER_CONSTANT)
            if x+delta > ecols or  y + delta  > erows:
                print(color)
                if  x+ delta > ecols:
                    blurry[y - delta:y + delta, x - delta:ecols, color] = convolved_image[y - delta:y + delta, x - delta:ecols]
                else:
                    blurry[y - delta:erows, x - delta:x + delta, color] = convolved_image[y - delta:erows, x - delta:x + delta]
            else:
                blurry[y - delta:y + delta, x - delta:x + delta, color] = convolved_image[y - delta:y + delta , x - delta:x + delta]
    blurry = 255*blurry
    blurry = blurry.astype(np.uint8)
    return blurry


def degeneration(img, psfs, device=torch.device('cuda'), Nx=7, Ny=7, psfsize=51,metadata =None):
    """ Degeneration for RGB images
    Args:
        img: (H, W, 3)
        psfs: [3, kernel_size, kernel_size, 64]
        Nx, Ny : The numbers of spatially varying PSFs in the horizontal and vertical directions.
        psfsize : kernel size
        metadata: ISP parameters, such as CCM matrix, gains..
    """
    # create the final rendered image
    # if isinstance(img,np.ndarray):
    #     img = torch.from_numpy(img).float()
    render_image = torch.zeros_like(img).to(device)

    # the size of original image
    H, W = img.shape[0], img.shape[1]

    # padding the original image
    pad = np.uint8((psfsize - 1) / 2)
    # img_pad = F.pad(img, (0, 0, pad, pad, pad, pad), 'constant', 0).to(device)
    img = img.permute(2, 0, 1)
    img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect').to(device).float()
    img_pad = img_pad/255
    img, metadata = unprocess_torch.unprocess_wo_mosaic(img_pad,metadata)
    img = img.cuda()

    # Divided into paths and convolved with their respective PSFs
    for i in range(Nx):
        for j in range(Ny):
            cur_psf = psfs[i,j,...].permute(2,1,0) # from H*W*C to C*H*W
            cur_psf = cur_psf.to(device).unsqueeze(1)
            h_low, w_low = int(i/Nx*H), int(j/Ny*W)
            h_high, w_high = int((i + 1) / Nx * H), int((j + 1) / Ny * W)
            cur_img = img[:, h_low:h_high + 2 * pad, w_low:w_high + 2 * pad]
            cur_img = cur_img.unsqueeze(0).float()
            # Separable Convolution per channel
            render_patch = F.conv2d(cur_img, cur_psf, groups=3, padding='valid', bias=None, stride=1)
            render_patch = render_patch.squeeze(0).permute(1, 2, 0)
            render_image[h_low:h_high, w_low:w_high] = render_patch

    bayer_image = unprocess_torch.mosaic(render_image)
    gain_bayer = process_torch.apply_gains(bayer_image, metadata['red_gain'], metadata['blue_gain'])
    img = process_torch.demosaic(gain_bayer)
    image1 = img.permute(1, 2, 0)
    # render_image = render_image
    # render_image = torch.clamp(render_image, min=0.0, max=1.0)
    image = process_torch.process(image1, metadata['red_gain'], metadata['blue_gain'], metadata['cam2rgb'])
    image = 255*image
    image = image.squeeze().to(torch.uint8)
    # plt.imshow(image)
    # plt.show()
    # B = image.squeeze().to(torch.uint8)
    return image,metadata

def patch_and_convolve(img, psfs, device=torch.device('cuda'), Nx=7, Ny=7, psfsize=51):
    """ Degeneration for RGB images
    Args:
        img: (H, W, 3)
        psfs: [3, kernel_size, kernel_size, 64]
        Nx, Ny : The numbers of spatially varying PSFs in the horizontal and vertical directions.
        psfsize : kernel size
        metadata: ISP parameters, such as CCM matrix, gains..
    """
    # create the final rendered image
    # if isinstance(img,np.ndarray):
    #     img = torch.from_numpy(img).float()
    render_image = torch.zeros_like(img).to(device)

    # the size of original image
    H, W = img.shape[0], img.shape[1]

    # padding the original image
    pad = np.uint8((psfsize - 1) / 2)
    # img_pad = F.pad(img, (0, 0, pad, pad, pad, pad), 'constant', 0).to(device)
    img = img.permute(2, 0, 1)
    img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect').to(device).float()
    img = img_pad.cuda()

    # Divided into paths and convolved with their respective PSFs
    for i in range(Nx):
        for j in range(Ny):
            cur_psf = psfs[i,j,...].permute(2,1,0) # from H*W*C to C*H*W
            cur_psf = cur_psf.to(device).unsqueeze(1)
            h_low, w_low = int(i/Nx*H), int(j/Ny*W)
            h_high, w_high = int((i + 1) / Nx * H), int((j + 1) / Ny * W)
            cur_img = img[:, h_low:h_high + 2 * pad, w_low:w_high + 2 * pad]
            cur_img = cur_img.unsqueeze(0).float()
            # Separable Convolution per channel
            render_patch = F.conv2d(cur_img, cur_psf, groups=3, padding='valid', bias=None, stride=1)
            render_patch = render_patch.squeeze(0).permute(1, 2, 0)
            render_image[h_low:h_high, w_low:w_high] = render_patch

    return render_image


def seidel2psf(seidel,IS,color):
        device = seidel.device
        BS = seidel.shape[0]
        # print(seidel.shape)
        M = IS.wf_res[color]
        sel_basis = IS.seidel_basis[color]
        num_seidel = sel_basis[color].shape[-1]
        seidel1 = seidel.unsqueeze(1).unsqueeze(1)
        # print(seidel.shape,seidel1.shape)
        seidel2 = seidel1.repeat(1, M, M, 1)
        WF = torch.zeros(BS,M, M).to(device)
        A = sel_basis.unsqueeze(0).repeat(BS, 1, 1, 1).to(device)
        # print(A.shape)
        # print(seidel2.shape)
        x = torch.linspace(-1, 1, M).to(device)
        Y, X = torch.meshgrid(x, x, indexing='ij')
        rho = torch.abs(torch.sqrt(X ** 2 + Y ** 2)).to(device)
        rho = rho.repeat(BS,1,1)
        for i in range(num_seidel):
            WF = WF + A[...,i] * seidel2[...,i]
        WF = torch.where(rho >= 1, 0, WF)
        M = WF.size(1)
        W = torch.nn.ZeroPad2d(2*M)(WF)
        phase = torch.exp(-1j * 2 * torch.pi * W)
        phase=torch.where(phase==1,0,phase)
        phase= fft2(phase)
        phase= fftshift(phase)
        AP = abs(phase) ** 2
        # AP = abs(fftshift(fft2(phase))) ** 2
        CenterCrop = torchvision.transforms.CenterCrop(M)
        AP = CenterCrop(AP)
        AP = AP / torch.max(AP)
        return AP

def tensor2img(tensor):
    """tensor to image for saving by opencv"""
    tensor = 255*tensor
    tensor = tensor.to(torch.uint8).numpy()
    img = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    return img

