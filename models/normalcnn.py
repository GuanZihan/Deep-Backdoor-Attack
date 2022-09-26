# -*- coding: utf-8 -*-
# @Data:2020/7/14 17:37
# @Author:lyg

from __future__ import absolute_import

import torch.nn as nn
import math
import torch.nn.functional as F
# __all__ = ['normalcnn']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # out += residual
        out = self.relu(out)
        

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out += residual
        
        out = self.relu(out)

        return out


class NormalCNN(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(NormalCNN, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)

        self.downsample2 = block(16, 32, 2, nn.Sequential(
                nn.Conv2d(16, 32 * block.expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(32 * block.expansion),
            ))
        self.downsample3 = block(32, 64, 2, nn.Sequential(
                nn.Conv2d(32, 64,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64),
            ))
        
        self.avgpool2 = nn.AvgPool2d(16)
        self.fc2 = nn.Linear(32 * block.expansion, num_classes)

        self.avgpool1 = nn.AvgPool2d(32)
        self.fc1 = nn.Linear(16 * block.expansion, num_classes)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        # self.fc3 = nn.Linear(3072, 64, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        output = block(self.inplanes, planes, stride, downsample)
        # print(self.inplanes, planes, stride, downsample)
        layers.append(output)
        
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            output = block(self.inplanes, planes)
            layers.append(output)

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        # # x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 16x16
    
        x = self.layer2(x)
        # activation1 = x
        # activation1 = self.downsample2(x)
        # activation1 = self.downsample3(activation1)
        # activation1 = self.avgpool3(x)
        # activation1 = activation1.view(activation1.size(0), -1)
        # activation1 = self.fc3(activation1)
        
        # x = self.layer2(x)  # 32x32
        # activation2 = x

        # activation2 = self.downsample3(x)
        # activation2 = self.avgpool2(activation2)
        # activation2 = activation2.view(activation2.size(0), -1)

        # activation2 = self.fc2(activation2)

        
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)        

        x = x.view(x.size(0), -1)
        # print(x.shape)
        # input()
        # print(x.shape)
        # input()
        x = self.fc(x)
        # x = self.fc2(x)

        return x


def normalcnn(**kwargs):
    """
    Constructs a ResNet model.
    """
    return NormalCNN(**kwargs)

# class NormalCNN(nn.Module):
#   def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
#     super(NormalCNN, self).__init__()
#     self.conv1 = nn.Conv2d(3, 6, 5)
#     self.pool = nn.MaxPool2d(2, 2)
#     self.conv2 = nn.Conv2d(6, 16, 5)
#     self.fc1 = nn.Linear(16 * 5 * 5, 120)
#     self.fc2 = nn.Linear(120, 84)
#     self.fc3 = nn.Linear(84, 10)
#   def forward(self, x):
#     x = self.pool(F.relu(self.conv1(x)))
#     x = self.pool(F.relu(self.conv2(x)))
#     x = x.view(-1, 16 * 5 * 5)
#     x = F.relu(self.fc1(x))
#     x = F.relu(self.fc2(x))
#     x = self.fc3(x)
#     return x