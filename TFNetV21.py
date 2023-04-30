import torch
from torch.nn import *
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os


class TFNet(nn.Module):  # 自定义的网络模板
    def __init__(self, Dim=[3, 34, 31], Depth=3, KS_1=3, KS_2=3, KS_3=3):  # 初始化函数
        super(TFNet, self).__init__()  # 调用某类的初始化方法

        self.lr_conv1 = nn.Sequential(
            nn.Conv2d(31, 32, kernel_size=3, stride=1,
                      padding=1),  # 31可改为输入的HS通道数
            nn.PReLU(),
        )
        self.lr_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.lr_down_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )
        self.hr_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1,
                      padding=1),  # 3可改为另外一个输入的其他图像的通道数
            nn.PReLU(),
        )
        self.hr_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.hr_down_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )

        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )

        self.recons_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.PReLU(),
        )
        self.recons_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 31, kernel_size=3, stride=1,
                      padding=1),  # 31可改为输入的HS通道数
            nn.PReLU(),
        )

    def forward(self, hrrgb, lrhsi):

        # feature extraction
        x_lr = self.lr_conv1(lrhsi)
        x_lr_cat = self.lr_conv2(x_lr)
        x_lr = self.lr_down_conv(x_lr_cat)

        x_hr = self.hr_conv1(hrrgb)
        x_hr_cat = self.hr_conv2(x_hr)
        x_hr = self.hr_down_conv(x_hr_cat)
        x = torch.cat((x_hr, x_lr), dim=1)

        # feature fusion
        x = self.fusion_conv1(x)
        x = torch.cat((x, self.fusion_conv2(x)), dim=1)

        # image reconstruction
        x = self.recons_conv1(x)
        x = torch.cat((x_lr_cat, x_hr_cat, x), dim=1)
        x = self.recons_conv2(x)

        return x


class TFNet_ori(nn.Module):
    def __init__(self):
        super(TFNet, self).__init__()
        self.encoder1_pan = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU())
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
        )
        self.restore1 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            nn.PReLU())
        self.restore2 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=2,
                               stride=2),
            nn.PReLU()
        )
        self.restore3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=4,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_pan, x_lr):
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        fusion1 = self.fusion1(torch.cat((encoder2_pan, encoder2_lr), dim=1))
        fusion2 = self.fusion2(fusion1)

        restore1 = self.restore1(fusion2)
        restore2 = self.restore2(torch.cat((restore1, fusion1), dim=1))
        restore3 = self.restore3(
            torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return restore3
