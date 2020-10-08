'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        return out

class ResNet(nn.Module):
    def __init__(self, block, input_channels=3, num_classes=10, channels=None):
        super(ResNet, self).__init__()

        channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
        
        #Preparation Layer
        self.prep_layer = nn.Sequential(
            nn.Conv2d(input_channels, channels['prep'], kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels['prep']),
            nn.ReLU()
        )

        # Layer1
        self.layer1 = nn.Sequential(
            nn.Conv2d(channels['prep'], channels['layer1'], kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(channels['layer1']),
            nn.ReLU(),
            block(channels['layer1'], channels['layer1'], stride=1)
        )

        # Layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels['layer1'], channels['layer2'], kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(channels['layer2']),
            nn.ReLU()
        )
        # Layer3
        self.layer3 = nn.Sequential(
            nn.Conv2d(channels['layer2'], channels['layer3'], kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(channels['layer3']),
            nn.ReLU(),
            block(channels['layer3'], channels['layer3'], stride=1),
        )
        
        #Output Layer
        self.output_layer = nn.Sequential(
            nn.MaxPool2d(4), # MaxPooling with Kernel Size 4
            nn.Flatten(),
            nn.Linear(channels['layer3'], num_classes, bias=False) ## TODO
        )

    def forward(self, x):
        out = self.prep_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output_layer(out)
        return F.log_softmax(out)


def ResNetDavidNet():
    return ResNet(BasicBlock)

def test():
    net = ResNetDavidNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
