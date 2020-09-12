from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.base_network_utils as nutils

class Cifar10Net(nn.Module):
     def __init__(self, drop_val=0):
         super(Cifar10Net, self).__init__()
         
         #Conv Block1
         self.conv_blk1 = nn.Sequential(
             nutils.Conv2d_BasicBlock(inC=3, outC=32, ksize=(3,3), padding=1, drop_val=drop_val), #RF=3x3,output size = 32x32x32
             nutils.Conv2d_BasicBlock(inC=32, outC=64, ksize=(3,3), padding=1, drop_val=drop_val), #RF=5x5,output size = 32x32x64
             nutils.Conv2d_BasicBlock(inC=64, outC=128, ksize=(3,3), padding=1, drop_val=drop_val) #RF=7x7,output size = 32x32x128
             ) 
         
         # Transition Layer
         self.conv_blk1_transition = nutils.Conv2d_TransistionBlock(128,32) #RF=8x8,output size = 16x16x32
         
         #Conv Block2
         self.conv_blk2 = nn.Sequential(
             nutils.Conv2d_BasicBlock(inC=32, outC=64, ksize=(3,3), padding=1, dilation=2, drop_val=drop_val), #RF=16x16,output size = 14x14x64
             nutils.Conv2d_BasicBlock(inC=64, outC=128, ksize=(3,3), padding=1, dilation=2, drop_val=drop_val) #RF=24x24,output size = 12x12x128
             ) 
         
         # Transition Layer
         self.conv_blk2_transition = nutils.Conv2d_TransistionBlock(128,32) #RF=26x26,output size = 6x6x32
         
         #Conv Block3
         self.conv_blk3 = nn.Sequential(
             nutils.Conv2d_BasicBlock(inC=32, outC=64, ksize=(3,3), padding=1, drop_val=drop_val), #RF=34x34,output size = 6x6x64
             nutils.Conv2d_BasicBlock(inC=64, outC=128, ksize=(3,3), padding=1, drop_val=drop_val) #RF=42x42,output size = 6x6x128
             ) 
         
         # Transition Layer
         self.conv_blk3_transition = nutils.Conv2d_TransistionBlock(128,32) #RF=46x46,output size = 3x3x32
         
         #Conv Block4
         self.conv_blk4 = nn.Sequential(
             nutils.Conv2d_DepthWiseSeperable_BasicBlock(inC=32, outC=64, ksize=(3,3), padding=1, drop_val=drop_val), #RF=62x62,output size = 3x3x64
             nutils.Conv2d_DepthWiseSeperable_BasicBlock(inC=64, outC=128, ksize=(3,3), padding=1, drop_val=drop_val) #RF=7x7,output size = 32x32x128
             ) 
         
         # Output Block
         self.output_block = nn.Sequential(
             nn.AvgPool2d(kernel_size=3),                                                            #RF=94x94,output size = 1x1x128
             nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1,1), padding=0,  bias=False)  #RF=94x94,output size = 1x1x10
        )   
        
     def forward(self, x):

        x = self.conv_blk1(x) # convolution block-1
        x = self.conv_blk1_transition(x)

        x = self.conv_blk2(x) # convolution block-2
        x = self.conv_blk2_transition(x)

        x = self.conv_blk3(x) # convolution block-3
        x = self.conv_blk3_transition(x)

        x = self.conv_blk4(x) # convolution block-4

        # output 
        x = self.output_block(x) # 

        # flatten the tensor so it can be passed to the dense layer afterward
        x = x.view(-1, 10)
        return F.log_softmax(x)
