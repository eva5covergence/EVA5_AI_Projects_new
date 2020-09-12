from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Conv2d_BasicBlock(nn.Module):
    def __init__(self, inC, outC, ksize, padding=0, dilation=1, drop_val=0):
        super(Conv2d_BasicBlock,self).__init__()

        self.drop_val = drop_val

        self.conv = nn.Sequential(           
            nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=ksize, padding=padding, dilation=dilation, bias=False), 
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )

        self.dropout = nn.Sequential(           
            nn.Dropout(self.drop_val)
        )

    def forward(self, x):
        x = self.conv(x)
        if(self.drop_val!=0):
          x = self.dropout(x)
        return x

class Conv2d_DepthWiseSeperable(nn.Module):
    def __init__(self, inC, outC, ksize, padding=0, dilation=1):
        super(Conv2d_DepthWiseSeperable,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inC, out_channels=inC, groups=inC, kernel_size=ksize, padding=padding,  dilation=dilation, bias=False), # depth wise convolution
            nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=(1,1), bias=False) # Pointwise convolution
        )

    def forward(self, x):
        return self.conv(x)

class Conv2d_DepthWiseSeperable_BasicBlock(nn.Module):
    def __init__(self, inC, outC, ksize, padding=0, dilation=1, drop_val=0):
        super(Conv2d_DepthWiseSeperable_BasicBlock,self).__init__()

        self.drop_val = drop_val

        self.conv = nn.Sequential(           
            Conv2d_DepthWiseSeperable(inC, outC, ksize, padding, dilation=dilation),   
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )

        self.dropout = nn.Sequential(           
            nn.Dropout(self.drop_val)
        )

    def forward(self, x):
        x = self.conv(x)
        if(self.drop_val!=0):
          x = self.dropout(x)
        return x

class Conv2d_TransistionBlock(nn.Module):
    def __init__(self, inC, outC):
        super(Conv2d_TransistionBlock,self).__init__()

        self.conv = nn.Sequential(           
            nn.MaxPool2d(2, 2),                                                           
            nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=(1, 1), bias=False),
        )

    def forward(self, x):
        return self.conv(x)