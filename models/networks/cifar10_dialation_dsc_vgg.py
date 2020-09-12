from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.base_network_utils as nutils

class Cifar10NetVGG(nn.Module):
     def __init__(self, drop_val=0):
         super(Cifar10NetVGG, self).__init__()
         
         #Conv Block1
         self.conv_blk1 = nn.Sequential(
             nutils.Conv2d_BasicBlock(inC=3, outC=32, ksize=(3,3), padding=1, drop_val=drop_val), #RF=3x3,output size = 32x32x16
             nutils.Conv2d_BasicBlock(inC=32, outC=32, ksize=(3,3), padding=1, drop_val=drop_val), #RF=5x5,output size = 32x32x32
             nutils.Conv2d_BasicBlock(inC=32, outC=32, ksize=(3,3), padding=1, drop_val=drop_val), #RF=7x7,output size = 32x32x64
             ) 
         
         self.pool1 = nn.MaxPool2d(2,2)
         
         #Conv Block2
         self.conv_blk2 = nn.Sequential(
             nutils.Conv2d_BasicBlock(inC=32, outC=64, ksize=(3,3), padding=2, dilation=2, drop_val=drop_val), #RF=16x16,output size = 14x14x32
             nutils.Conv2d_BasicBlock(inC=64, outC=64, ksize=(3,3), padding=2, dilation=2, drop_val=drop_val) #RF=24x24,output size = 12x12x64
             ) 
         
         self.pool2 = nn.MaxPool2d(2,2)
         
         #Conv Block3
         self.conv_blk3 = nn.Sequential(
             nutils.Conv2d_BasicBlock(inC=64, outC=128, ksize=(3,3), padding=1, drop_val=drop_val), #RF=34x34,output size = 6x6x32
             nutils.Conv2d_BasicBlock(inC=128, outC=128, ksize=(3,3), padding=1, drop_val=drop_val) #RF=42x42,output size = 6x6x64
             ) 
         
         self.pool3 = nn.MaxPool2d(2,2)
         
         #Conv Block4
         self.conv_blk4 = nn.Sequential(
             nutils.Conv2d_DepthWiseSeperable_BasicBlock(inC=128, outC=256, ksize=(3,3), padding=1, drop_val=drop_val), #RF=62x62,output size = 3x3x32
             nutils.Conv2d_DepthWiseSeperable_BasicBlock(inC=256, outC=256, ksize=(3,3), padding=1, drop_val=drop_val) #RF=78x78,output size = 3x3x64
             ) 
         
         # Output Block
         self.output_block = nn.Sequential(
             nn.AvgPool2d(kernel_size=4),                                                            #RF=94x94,output size = 1x1x64
             nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1,1), padding=0,  bias=False),  #RF=94x94,output size = 1x1x10
        )   
        
     def forward(self, x):

        x = self.conv_blk1(x) # convolution block-1
        x = self.pool1(x)

        x = self.conv_blk2(x) # convolution block-2
        x = self.pool2(x)

        x = self.conv_blk3(x) # convolution block-3
        x = self.pool3(x)

        x = self.conv_blk4(x) # convolution block-4

        # output 
        x = self.output_block(x) # 

        # flatten the tensor so it can be passed to the dense layer afterward
        x = x.view(-1, 10)
        return F.log_softmax(x)