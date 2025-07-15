import numpy as np

# 加载文件
data = np.load(r'E:\PSF-Estimation\dataset\63762BB\npy\x59_y-1337_fov9.44_angle182.47_v.npz')  # 替换为你的实际路径（npy_path）
data2 = np.load(r'E:\PSF-Estimation\dataset\63762BB_1\npy\x59_y-1337_fov9.44_angle182.47_v.npz')

# 查看所有保存的键名
print(data.files)  # 输出: ['sfr', 'weight', 'rot', 'fov', 'offset']

# 提取每个变量（按保存时的键名）
sfr = data['sfr']
weight = data['weight']
rot = data['rot']
fov = data['fov']
offset = data['offset']


sfr2 = data2['sfr']
weight2 = data2['weight']
rot2 = data2['rot']
fov2 = data2['fov']
offset2 = data2['offset']


# 验证数据形状（可选）
print(sfr.shape, weight.shape)  # 检查数组维度
print(sfr[:,1])
print(sfr2[:,1])
print(weight,weight2)