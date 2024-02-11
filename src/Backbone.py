import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = FrozenBatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = FrozenBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                FrozenBatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x 
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )

        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride = 2),
            BasicBlock(128, 128)
        )

        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride = 2),
            BasicBlock(256, 256)
        )

        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride = 2),
            BasicBlock(512, 512)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5