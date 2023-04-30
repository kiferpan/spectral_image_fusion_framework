import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
from scipy import io as sio
from skimage.util.shape import view_as_windows

import numpy as np 

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"]) # 判断文件是否是.mat


def load_image(filepath):
    img = Image.open(filepath)
    return img


def nomalize(img_data):
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
    return img_data

def get_block(img_data, window_size, band, step = 1):
    block = view_as_windows(img_data, [band, window_size, window_size], step = step)
    return block

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, device, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames =  [join(image_dir, x.split('_')[0]).replace('\\', '/')
                                for x in listdir(image_dir) if is_image_file(x)]
        # self.image_filenames = [] 
        self.image_dir = image_dir
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.device = device

    def __getitem__(self, index): # 数据的读取&处理

        pan_patches= []
        hsi_gt_patches = []
        lr_hsi_patches = []
        pan_patches, hsi_gt_patches, lr_hsi_patches = np.zeros((1, 1, 4, 4)), np.zeros((1, 4, 4, 4)), np.zeros((1, 4, 1, 1)) # 目标图像的尺寸大小
        print(self.image_filenames)

        for i in self.image_filenames:
            print(i)
            data = sio.loadmat(i) # 按照编号，读取.mat文件

            pan = np.array(data['pan'][...], dtype=np.float64) # 输入pan和hsi的尺寸应该为4:1的关系
            if(len(pan.shape) is 2): # 如果输入的图像为全色图像
                pan = np.expand_dims(pan, axis = 0 )
                pan = pan[:, ::4,::4] # 将全色图像下采样 需要写一个函数
            else: pan = pan[::4,::4]
            pan = nomalize(pan)

            hsi_gt = lr_hsi = np.array(data['ms'][...], dtype=np.float64)
            hsi_gt = nomalize(hsi_gt.reshape(hsi_gt.shape[-1], hsi_gt.shape[0], hsi_gt.shape[1]))
            print(hsi_gt.shape)

            lr_hsi = lr_hsi[::4,::4, :] # 从原始hsi图像中进行四倍缩小
            lr_hsi = nomalize(lr_hsi.reshape(lr_hsi.shape[-1], 
                        lr_hsi.shape[0], 
                        lr_hsi.shape[1])) + np.random.normal(0, 0.001, lr_hsi.reshape(lr_hsi.shape[-1], lr_hsi.shape[0], lr_hsi.shape[1]).shape)  # 生成高斯噪声，并加入到lrhsi中
            print(lr_hsi.shape)

            # pan_patches.append(get_block(pan, window_size = 4, step = 4, band = 1))
            # hsi_gt_patches.append( get_block(hsi_gt, window_size = 4, step = 4, band = 4))
            # lr_hsi_patches.append(get_block(lr_hsi, window_size = 1, step = 1, band = 4))

            pan = get_block( pan, 
                    window_size = 4, 
                    step = 4, 
                    band = 1)
            hsi_gt = get_block( hsi_gt, 
                    window_size = 4, 
                    step = 4, 
                    band = 4)
            lr_hsi = get_block( lr_hsi, 
                    window_size = 1, 
                    step = 1, 
                    band = 4)

            pan_patches = np.concatenate((pan_patches,
                            pan.reshape(-1, pan.shape[-3], pan.shape[-2], pan.shape[-1])),
                            axis=0)
            lr_hsi_patches = np.concatenate((lr_hsi_patches,
                            lr_hsi.reshape(-1, lr_hsi.shape[-3], lr_hsi.shape[-2], lr_hsi.shape[-1])),
                            axis=0)
            hsi_gt_patches = np.concatenate((hsi_gt_patches,
                            hsi_gt.reshape(-1, hsi_gt.shape[-3], hsi_gt.shape[-2], hsi_gt.shape[-1])),
                            axis=0)
        

        # 转张量
        get_dataset = data.TensorDataset(torch.tensor(pan_patches[1:,:,:,:], dtype=torch.float, device=self.device), 
                                                        torch.tensor(lr_hsi_patches[1:,:,:,:], dtype=torch.float, device=self.device), 
                                                        torch.tensor(hsi_gt_patches[1:,:,:,:], dtype=torch.float, device=self.device))
        # print(pan_patches.shape)
        # print(lr_hsi_patches.shape)
        # print(hsi_gt_patches.shape)
        # data = sio.loadmat('%s.mat' % self.image_filenames[index]) # 按照编号，读取.mat文件
        
        # print('%s is loaded' % self.image_filenames[index])
        # pan = np.array(data['pan'][...], dtype=np.float64) # 输入pan和hsi的尺寸应该为4:1的关系
        # pan = pan[::4,::4] # 将全色图像下采样,以保证和ms_gt的尺寸相当
        # pan = nomalize(pan)

        # hsi_gt = np.array(data['ms'][...], dtype=np.float64)
        # hsi_gt = nomalize(hsi_gt)

        # lr_hsi = hsi_gt[::4,::4, :] # 从原始hsi图像中进行四倍缩小
        # noise = np.random.normal(0, 0.001, lrhsi.shape) 
        # lr_hsi = nomalize(lr_hsi) + noise # 生成高斯噪声，并加入到lrhsi中


        # pan = get_block(pan, 4)
        # hsi_gt = get_block(hsi, 4)
        # lr_hsi = get_block(lr_hsi, 1)


        # input_pan = load_image('%s_pan.tif' % self.image_filenames[index])
        # input_lr = load_image('%s_lr.tif' % self.image_filenames[index])
        # input_lr_u = load_image('%s_lr_u.tif' % self.image_filenames[index])
        # target = load_image('%s_mul.tif' % self.image_filenames[index])


        # filename = int(self.image_filenames[index].split('/')[-1])
        # if self.input_transform:
        #     pan = self.input_transform(pan)
        #     lr_hsi = self.input_transform(lr_hsi)
        # if self.target_transform:
        #     hsi_gt = self.target_transform(hsi_gt)

        return get_dataset # pan_patches, lr_hsi_patches, hsi_gt_patches # , filename
    
    # def __len__(self):
    #     return 0
    #     return len(self.image_filenames)
