import os
import glob
import numpy as np
import scipy.io as scio
from scipy.io import loadmat ,savemat
import argparse
import yaml
import torch
import utils.tools
import model.optics_rgb
import math
import matplotlib.pyplot as plt

def config(path):
    """ Config file for training.
    """
    # Config file
    current_path = os.getcwd()
    with open(path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args['in_path']= os.path.join(current_path,args['input_dir'],args['filename']+'.xlsx')
    list = ['mat', 'npy', 'crop']

    for key in list:
        path = os.path.join(args['dataset'],args['filename'], key)
        args[key] = path
        if not os.path.exists(path):
            os.makedirs(path)
    return args

def save_npy(gt_list,weights, filename_l, path):
    lens = len(gt_list)
    width = (len(gt_list[0]['sfr'][:, 1]) + 1) // 2

    print(width)

    for img_idx in range(lens):
        filename = os.path.basename(filename_l[img_idx])
        npy_path = os.path.join(path,filename[:-4])
        # red channel(0), green channel(1), blue channel(2)
        sfr= gt_list[img_idx]['sfr'][:,1:4][0:width]
        # sfr_g = gt_list[img_idx]['sfr'][:,2][0:width]
        # sfr_b = gt_list[img_idx]['sfr'][:,3][0:width]
        weight = weights[img_idx]
        rot = gt_list[img_idx]['rot']
        fov = gt_list[img_idx]['fov']
        offset = gt_list[img_idx]['offset']
        # if fov > 35:
        #     print(rot)
        #     print(offset)
        # dict ={'sfr':sfr,'weight':weight,'rot':rot,'fov':fov}
        np.savez(npy_path,sfr=sfr,weight=weight,rot=rot,fov=fov,offset=offset)


if __name__ == "__main__":
    path = 'configs/ss.yaml'
    args = config(path)
    print(args['npy'])
    IS = model.optics_rgb.IS(filepath=args['in_path'])
    hfov = int(IS.hfov)+ 1
    # directory = args['mat']
    directory = '.\dataset\\63762BB\mat\shot0.00'
    # arg['npy'] =
    mat_files = [file for file in os.listdir(directory) if file.endswith('.mat')]
    sorted_mat_files = sorted(mat_files)
    # hfov = args['hfov']

    data = []
    filename = []
    # 遍历文件夹中的所有文件
    for file_name in sorted_mat_files:
        file_path = os.path.join(directory, file_name)
        # 使用loadmat函数读取.mat文件
        temp = loadmat(file_path)
        data.append(temp)
        filename.append(file_name)

    # length = (temp['sfr'].shape[0]+1)//2
    data = [{'rot': item['rot'] + 360 if item['rot'] < 0 else item['rot'],'sfr': item['sfr'], 'fov': item['fov'],'offset': item['offset']} for item in data]

    'determine weight matrix'
    fov_m, weights_all, gt_list ,filename_l = [], [], [],[]
    for i in range(hfov-1):
        # filter all the mtf belong [i,i+1)
        filtered_list = [item for item in data if i <= item['fov'] < i + 1]
        filtered_list = sorted(filtered_list, key=lambda x: x['rot'])
        sel_file = [filename[index] for index, item in enumerate(data) if i <= item['fov'] < i + 1]
        filename_l.append(sel_file)
        gt_list.append(filtered_list)
        fov_m.append(len(filtered_list))

        angles = np.array([item['rot'] for item in filtered_list], dtype=np.float64)

        angles = torch.tensor(angles)
        # fov_weight = math.sin(math.radians((i * 80 / hfov + 10)))
        fov_weight = 1
        weights_m = utils.tools.weights(angles, bins=8)
        weights_all.append(weights_m*fov_weight)


    weights_all1 = torch.cat(weights_all)
    weights_all1 = weights_all1/torch.sum(weights_all1)
    gt_list = [item for sublist in gt_list for item in sublist]
    filename_l = [item for sublist in filename_l for item in sublist]
    save_npy(gt_list, weights_all1, filename_l, args['npy'])









