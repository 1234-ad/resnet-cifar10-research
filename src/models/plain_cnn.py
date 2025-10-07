"""
Plain CNN Implementation for CIFAR-10
Equivalent architecture to ResNet but WITHOUT skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainBlock(nn.Module):
    """Plain block without skip connections"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # NO skip connection here - this is the key difference
        out = F.relu(out)
        return out


class PlainCNN(nn.Module):
    """Plain CNN architecture (ResNet without skip connections)"""
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(PlainCNN, self).__init__()
        self.in_planes = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Plain layers (no skip connections)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final classifier
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PlainCNN18(num_classes=10):
    """Plain CNN-18 (ResNet-18 without skip connections)"""
    return PlainCNN(PlainBlock, [2, 2, 2, 2], num_classes)


def PlainCNN34(num_classes=10):
    """Plain CNN-34 (ResNet-34 without skip connections)"""
    return PlainCNN(PlainBlock, [3, 4, 6, 3], num_classes)


# Test function
if __name__ == "__main__":
    net = PlainCNN18()
    y = net(torch.randn(1, 3, 32, 32))
    print(f"Output shape: {y.size()}")
    print(f"Number of parameters: {sum(p.numel() for p in net.parameters())}")