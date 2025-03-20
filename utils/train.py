import sys
import torch.nn.functional as F
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import tools
from datetime import datetime
import cv2

def config(source):
    """ Config file
    """
    current_path = os.getcwd()
    with open(source) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    filename = args['filename']
    args['in_path'] = os.path.join(current_path, args['input_dir'], filename + '.xlsx')
    args['result_path'] = os.path.join(current_path, args['result'], filename)
    return args


def rotatepsf(psf, angle):
    """
    Rotate PSF pattern to align with checkerboard location.
    - input: psf:[N C H W] theta[N 2 3]
    - output: psf [N C H W]
    """
    device = psf.device
    rot = torch.tensor(angle).to(device)
    if len(psf.shape)==2:
        psf = psf.unsqueeze(0).unsqueeze(0)
    elif len(psf.shape)==3:
        psf = psf.unsqueeze(1)
    if len(rot.shape) != 0:
        BS = rot.shape[0]
        if psf.shape[0] ==1:
            psf = psf.repeat(BS,1,1,1)
    else:
        BS = psf.shape[0]
    theta = torch.zeros((BS, 2, 3), dtype=torch.float).to(device)
    theta[:, 0, 0] = torch.cos(rot)
    theta[:, 0, 1] = -torch.sin(rot)
    theta[:, 1, 0] = torch.sin(rot)
    theta[:, 1, 1] = torch.cos(rot)
    grid = F.affine_grid(theta, size=psf.size()).to(device)
    rotated_psf = F.grid_sample(psf.float(),grid.float())
    rotated_psf = rotated_psf.squeeze()/ torch.sum(rotated_psf)
    return rotated_psf


def psf_map(psfs):

    'input: psf list, output: psf_map'

    X = 9
    Y = 12
    num = len(psfs)
    Hs = torch.linspace(0, 1, num)
    channel = psfs[0].size(2)
    if len(psfs[0].size()) == 3:
        [h, w, c] = psfs[0].size()
    else:
        [h, w] = psfs[0].size()
        c = 1
    psf_list = []
    for i in range(X):
        for j in range(Y):
            cy, cx = ((X - 1) / 2 - i), (j - (Y - 1) / 2)
            H = np.sqrt(cx ** 2 + cy ** 2) / np.sqrt(((X - 1) / 2) ** 2 + ((Y - 1) / 2) ** 2)
            index = torch.argmin(torch.abs(H - Hs))
            rot = np.arctan2(cy, cx) - np.pi / 2
            psf = []
            for color in range(channel):
                psf_single = psfs[index][:, :, color]
                rot_psf = rotatepsf(psf_single, rot)
                psf.append(rot_psf)
            psf = torch.stack(psf, dim=-1)
            psf_list.append(psf)
    psf_matrix = torch.stack(psf_list).view(X, Y, h, w, c)
    psfmap = psf_matrix.permute(0, 2, 1, 3, 4).contiguous().view(X * h, Y * w, c)
    return psf_matrix, psfmap

def shiftpsf(x, shifts, mode='bilinear'):
    if len(x.squeeze().shape) == 2:
        x = x.squeeze().unsqueeze(0).unsqueeze(0)
    elif len(x.squeeze().shape) == 3:
        x = x.squeeze().unsqueeze(1)
    batch_size, channels, height, width = x.shape
    theta = torch.zeros(batch_size, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = shifts[0]/height
    theta[:, 1, 2] = shifts[1]/height
    # Generate grid
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    # Apply grid sampling (bilinear interpolation by default)
    x_out = F.grid_sample(x, grid, mode=mode, align_corners=False)
    return x_out.squeeze()

def rot_interg(psf,rot):
    psf = rotatepsf(psf, rot)
    # input: psf [N,h,w], output: normalized intergrated area:[N]
    lsf = torch.sum(psf, dim=1)
    esf = torch.cumsum(lsf, dim = -1)
    nor_esf = esf/torch.max(esf)
    interg = torch.sum(nor_esf,dim=-1)
    return interg


def train(net,shiftnet, IS, args):
    lr = args['lr']
    epochs = args['epochs']
    interval = args['interval']
    # net.init_weights()

    folder_path = glob.glob(os.path.join(args['npy'], '*.npz'))
    optimizer = torch.optim.AdamW(net.parameters(), lr)

    keys = ['sfr', 'weight', 'fov', 'rot', 'offset']
    data_dict = {key: [] for key in keys}

    for file in folder_path:
        data = np.load(file)
        for key in keys:
            tensor = torch.from_numpy(data[key].astype(np.float32) if key == 'rot' else data[key]).cuda()
            data_dict[key].append(tensor)

    sfrs, weights, fovs, rots, offsets = [torch.stack(data_dict[key], dim=-1).squeeze() for key in keys]

    Hs = torch.linspace(0,1,21)
    psf_all = []
    colors = ['R','G','B']
    for color in range(3):
        psfs , H_all, i_H = [], [], 0
        for angle in range(1,IS.hfov + interval,interval):

            coe = 0.51
            id = torch.where((fovs >= (angle-coe*interval)) & (fovs <= (angle + coe*interval)))[0]
            weight, rot, fov = (tensor[id] for tensor in [weights, rots, fovs])
            H = tools.fov2H(fov, IS)
            sfr = sfrs[...,color,id]
            IS.seidel_basis = IS.s_basis(IS.wf_res, type='ss')

            for epoch in range(epochs):
                _, _, psf = net(IS, H, color)
                psf = tools.downsample(psf, IS.s_psf)
                mtf_2d = tools.PSF2MTF(psf, size=(2 * IS.s_psf - 1) * 5)
                mtf = tools.slice(mtf_2d, rot).cuda()
                loss = torch.sum(weight * torch.mean(torch.abs(mtf - sfr.T), dim=-1))
                optimizer.zero_grad()
                loss.backward()
                del mtf_2d, mtf
                optimizer.step()
                if epoch % 500 == 0 and epoch > 0 :
                    print('Optimize PSF lens:{} @ fov:{}, channel:{}, epoch={}, loss = {}'.format(args['filename'], angle, colors[color], epoch, loss.cpu().detach().numpy()))
            psf_l = [psf[i,...].detach().cpu() for i in range(psf.shape[0])]
            H_l = [H[i] for i in range(H.shape[0])]
            psfs= psfs + psf_l
            H_all = H_all + H_l
        H_all = torch.tensor(H_all)

        for i in range(len(Hs)):
            H = Hs[i]
            # Identify the closest normalized field height and retrieve the corresponding PSF.
            idx = torch.argmin(torch.abs(H_all-H))
            psf_all.append(psfs[idx])

    rpsfs, gpsfs ,bpsfs= [psf_all[i*len(Hs):(i+1)*len(Hs)] for i in range(3)]
    psf_ori = [torch.stack([rpsfs[i],gpsfs[i] ,bpsfs[i]],dim=-1) for i in range(len(Hs))]

    # optimize chromatic aberration
    optimizer = torch.optim.AdamW(shiftnet.parameters(), lr)

    psf_final = []
    for index in range(len(Hs)):
        H = Hs[index]
        angle = tools.H2fov(H,IS)
        id = torch.where((fovs >= (angle - 1)) & (fovs < (angle + 1)))[0]
        rpsf, gpsf, bpsf = rpsfs[index].squeeze().detach(), gpsfs[index].squeeze().detach(), bpsfs[index].squeeze().detach()
        weight, rot,  offset = weights[id], rots[id], offsets[...,id]
        for epoch in range(epochs):
            shift = shiftnet(H).squeeze()
            rpsf_s = shiftpsf(rpsf, shift[0:2])
            bpsf_s = shiftpsf(bpsf, shift[2:])
            psfs = [rpsf_s,gpsf,bpsf_s]
            interg = []
            for i in range(3):
                interg.append(rot_interg(psfs[i], rot))
            shiftr = interg[0] - interg[1]
            shiftb = interg[2]-  interg[1]
            loss = torch.sum(weight * (torch.abs(shiftr.cuda() - offset[0, :]) + torch.abs(shiftb.cuda() - offset[1, :])))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        psf_final.append(torch.stack([rpsf_s,gpsf,bpsf_s],dim=-1))


    # create directory for restore results
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    result_dir = '{}'.format(current_time)
    result_dir = os.path.join(args['result_path'], result_dir)
    print(f'Results saved in: {result_dir}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # save results, psf_matrix (9,12,25,25,3) psfmap [225,300,3]
    psf_matrix, psfmap = psf_map(psf_ori)
    np_save_path = os.path.join(result_dir, 'psf.npy')
    np.save(np_save_path,{'psf_matrix': psf_matrix.detach().cpu().numpy(), 'psfmap': psfmap.detach().cpu().numpy(), 'psfs': psf_ori})
    showmap = psfmap/torch.max(psfmap)
    plt.imshow(showmap.detach().cpu().numpy())
    plt.savefig(os.path.join(result_dir, 'psfmap_ori.png'))

    psf_matrix, psfmap = psf_map(psf_final)
    np_save_path = os.path.join(result_dir, 'psf_after_shift.npy')
    np.save(np_save_path,{'psf_matrix': psf_matrix.detach().cpu().numpy(), 'psfmap': psfmap.detach().cpu().numpy(), 'psfs': psf_final})
    showmap = psfmap/torch.max(psfmap)
    plt.imshow(showmap.detach().cpu().numpy())
    plt.savefig(os.path.join(result_dir, 'psfmap_shift.png'))

    gt_psfs = IS.Zer2PSF2(IS, Num=10)
    _, gt_psfmap = psf_map(gt_psfs)

    angle = np.rad2deg(np.arctan(IS.res[0] / IS.res[1]))
    psf_show  = []
    num = 6
    for i in range(num):
        psf_l = []
        gt_i = int(i * (len(gt_psfs) - 1) / 5)
        fit_i = int(i * (len(Hs) - 1) / 5)
        psf_l.extend([gt_psfs[gt_i],psf_ori[fit_i],psf_final[fit_i]])

        for i in range(3):
            psf = rotatepsf(psf_l[i].permute(2,0,1), angle)
            psf = psf.permute(1,2,0)
            psf = psf / torch.amax(psf, dim=(0, 1), keepdim=True)
            psf_show.append(psf)

    psf_matrix = torch.stack(psf_show).view(num,3, 25, 25, 3)
    psfmap = psf_matrix.permute(0, 2, 1, 3, 4).contiguous().view(num * 25, 3 * 25, 3).transpose(0,1)
    img = tools.tensor2img(psfmap)
    img_save_path = os.path.join(result_dir, 'compare.png')
    cv2.imwrite(img_save_path, img)



