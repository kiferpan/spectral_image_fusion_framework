import numpy as np
from os import listdir
from os.path import join, basename, exists
from PIL import Image
import torch
import torch.utils.data as Data

from scipy import io as sio
from skimage.util.shape import view_as_windows

from transforms import Stretch
from torchvision.transforms import Compose, ToTensor

# 舍弃
# from dataset import DatasetFromFolder

# 判断文件是否是.mat
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"]) 

# 加载数据
def load_image(filepath):
    img = Image.open(filepath)
    return img

# 数据归一化
def nomalize(img_data):
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
    return img_data

# img_data:输入图像数据，以numpy数组
# window_size:图像分割大小, 对数据集做分割处理
# band:数据集波段数
# step:滑动窗口步伐，默认1
def get_block(img_data, window_size, band, step = 1):
    block = view_as_windows(img_data, [band, window_size, window_size], step = step)
    return block

# 数据增强
def input_transform():
    return Compose([ToTensor(), Stretch()])

# 数据增强
def target_transform():
    return Compose([ToTensor(), Stretch()])

# 获得图像文件列表
def image_filenames(image_dir):
    return [join(image_dir, x.split('_')[0]).replace('\\', '/')
                                for x in listdir(image_dir) if is_image_file(x)]

# 制作数据集
# 参数：
# dir:数据集路径
# device:GPU参数
# 
# 输入数据集要求: 
# 1. 以'数字+.mat'文件保存
# 2. .mat中包含'ms'和'pan'两组数据,且size以4的倍数保存(方便下采样后，数据前后尺寸一致) 

# 如果需要修改网络训练数据集大小，则首先需要修改 pan_patches, hsi_gt_patches, lr_hsi_patches 三个patches
# 的patches[2],patches[3]
def get_dataset(dir, device):
    pan_patches= []
    hsi_gt_patches = []
    lr_hsi_patches = []
    pan_patches, hsi_gt_patches, lr_hsi_patches = np.zeros((1, 3, 16, 16)), np.zeros((1, 31, 16, 16)), np.zeros((1, 31, 4, 4)) # 目标图像的尺寸大小
    
    # print(self.image_filenames)

    for i in image_filenames(dir):
        data = sio.loadmat(i) # 按照编号，读取.mat文件

        pan = np.array(data['pan'][...], dtype=np.float64) # 输入pan和hsi的尺寸应该为4:1的关系
        if(len(pan.shape) is 2): # 如果输入的图像为全色图像
            pan = np.expand_dims(pan, axis = 0 )
            pan = pan[:, ::4,::4] # 将全色图像下采样 
        else:  # 如果输入图像是RGB图像
            pan = pan.reshape(pan.shape[-1], pan.shape[-3], pan.shape[-2])
            pan = pan[:, ::4, ::4]
        pan = nomalize(pan)

        hsi_gt = lr_hsi = np.array(data['ms'][...], dtype=np.float64) 
        hsi_gt = nomalize(hsi_gt.reshape(hsi_gt.shape[-1], hsi_gt.shape[0], hsi_gt.shape[1]))

        lr_hsi = lr_hsi[::4,::4, :] # 从原始hsi图像中进行四倍缩小
        lr_hsi = nomalize(lr_hsi.reshape(lr_hsi.shape[-1], 
                    lr_hsi.shape[0], 
                    lr_hsi.shape[1])) + np.random.normal(0, 0.001, lr_hsi.reshape(lr_hsi.shape[-1], lr_hsi.shape[0], lr_hsi.shape[1]).shape)  # 生成高斯噪声，并加入到lrhsi中

        pan = get_block( pan, 
                window_size = 16, 
                step = 4, 
                band = pan.shape[0])
        hsi_gt = get_block( hsi_gt, 
                window_size = 16, 
                step = 4, 
                band = hsi_gt.shape[0])
        lr_hsi = get_block( lr_hsi, 
                window_size = 4,  
                step = 1, 
                band = lr_hsi.shape[0])

        pan_patches = np.concatenate((pan_patches,
                        pan.reshape(-1, pan.shape[-3], pan.shape[-2], pan.shape[-1])),
                        axis=0)
        lr_hsi_patches = np.concatenate((lr_hsi_patches,
                        lr_hsi.reshape(-1, lr_hsi.shape[-3], lr_hsi.shape[-2], lr_hsi.shape[-1])),
                        axis=0)
        hsi_gt_patches = np.concatenate((hsi_gt_patches,
                        hsi_gt.reshape(-1, hsi_gt.shape[-3], hsi_gt.shape[-2], hsi_gt.shape[-1])),
                        axis=0)

    # 转张量, 并生成数据集
    get_final_dataset = Data.TensorDataset( torch.tensor(pan_patches[1:,:,:,:], dtype=torch.float, device=device), 
                                            torch.tensor(lr_hsi_patches[1:,:,:,:], dtype=torch.float, device=device), 
                                            torch.tensor(hsi_gt_patches[1:,:,:,:], dtype=torch.float, device=device))

    return get_final_dataset


def get_training_set(root_dir, device):
    train_dir = join(root_dir, "train")
    train_dir = train_dir.replace('\\', '/')
    return get_dataset(train_dir, device)
    
    # return DatasetFromFolder(train_dir,
    #                          device,
    #                          input_transform=input_transform(),
    #                          target_transform=target_transform())


def get_test_set(root_dir, device):
    test_dir = join(root_dir, "test")
    test_dir = test_dir.replace('\\', '/')
    return get_dataset(test_dir, device)
 
    # return DatasetFromFolder(test_dir,
    #                          device,
    #                          input_transform=input_transform(),
    #                          target_transform=target_transform())
