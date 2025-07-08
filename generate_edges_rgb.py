import os
import yaml
import model.optics_rgb
import model.checkerboard_rgb
import shutil
import cv2

def config(path):
    """ Config file for training.
    """
    # Config file
    current_path = os.getcwd()
    with open(path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args['in_path']= os.path.join(current_path,args['input_dir'],args['filename']+'.xlsx')
    args['in_path'] = os.path.join(current_path, args['input_dir'], args['filename'] + '.xlsx')
    list = ['mat','crop','npy']
    for key in list:
        path = os.path.join(args['dataset'],args['filename'], key, args['noise'])
        args[key] = path
        if not os.path.exists(path):
            os.makedirs(path)
    return args

def clear_create_folder(folder_path,clear=False):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在。")
        os.makedirs(folder_path)
        return

    # 删除文件夹中的所有文件和子文件夹
    if clear:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除文件夹及其内容
            except Exception as e:
                print(f'删除 {file_path} 时出错: {e}')


if __name__=='__main__':
    path = 'configs/ss.yaml'
    args = config(path)
    IS = model.optics_rgb.IS(filepath=args['in_path'])
    checker = model.checkerboard_rgb.checker(square_size = 200)
    checker.square_size = 200
    checker.latent(IS)

    save_path, mat_path = args['crop'],args['mat']
    clear_create_folder(save_path,clear= True)
    clear_create_folder(mat_path, clear=True)
    checker.crop(IS, save_path)










