import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # output_size = 32
        
        #Point wise convolution to increase the number of channel from 3 to 32 to support addition
        self.pconvblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(32)
        ) # output_size = 32
 

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # output_size = 32



        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        ) # output_size = 16

 
        #Point wise convolution to increase the number of channel from 32 to 64 to support addition
        self.pconvblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(64)
        ) # output_size = 16


        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        ) # output_size = 16
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        ) # output_size = 16



        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        ) # output_size = 8

        #Point wise convolution to increase the number of channel from 64 to 128 to support addition
        self.pconvblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128)
        ) # output_size = 8

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        ) # output_size = 8
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        ) # output_size = 8
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # output_size = 1

        self.fc = nn.Linear(in_features = 128, out_features = 10, bias=False)

 
    def forward(self, x1):
        x2 = self.convblock1(x1)
        x3 = self.convblock2(self.pconvblock1(x1)+x2)
        x4 = self.pool1(self.pconvblock1(x1)+x2+x3)  ### 32 channels
        x5 = self.convblock3(x4)
        x6 = self.convblock4(self.pconvblock2(x4)+x5) 
        x7 = self.convblock5(self.pconvblock2(x4)+x5+x6)  ### 64 channels
        x8 = self.pool2(x5+x6+x7)
        x9 = self.convblock6(x8)   ### 128 channels
        x10 = self.convblock7(self.pconvblock3(x8)+x9)
        x11 = self.convblock8(self.pconvblock3(x8) + x9 + x10)
        x12 = self.gap(x11)
        x12= x12.view(-1, 128)
        x13 = self.fc(x12)
        x14= x13.view(-1, 10)
        return F.log_softmax(x14, dim=-1)
