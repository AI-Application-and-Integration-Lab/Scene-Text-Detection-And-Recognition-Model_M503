import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
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

        out += residual
        out = self.relu(out)

        return out


class Conv_block(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._conv = torch.nn.Conv2d(*args, **kwargs)
    def forward(self, input):
        return torch.nn.functional.leaky_relu(self._conv(input),negative_slope=0.2)


# class Conv_block_bup(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self._conv = torch.nn.Conv2d(*args, **kwargs)
#         # nn.Conv2d(fpn_dim, fpn_dim, 3, 2, 1, bias=True),
#         self.gn = torch.nn.GroupNorm(4, 32, eps=1e-5)
#         # self.leaky = torch.nn.LeakyReLU(inplace=True)
#     def forward(self, input):
#         output=self._conv(input)
#         output=self.gn(output)
#         return torch.nn.functional.leaky_relu(output, negative_slope=0.2)


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        # self._conv1_1 = Conv_block(in_channels=512, out_channels=256, kernel_size = 3, stride =1, padding = 1)
        # self._conv2_1 = Conv_block(in_channels=256, out_channels=128, kernel_size = 3, stride =1, padding = 1)
        # self._deconv1 = torch.nn.ConvTranspose2d(128, 128, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        # self._conv3_1 = Conv_block(in_channels=128, out_channels=64, kernel_size = 3, stride =1, padding = 1)
        # self._conv4_1 = Conv_block(in_channels=64, out_channels=32, kernel_size = 3, stride =1, padding = 1) 

        # self._conv5_1 = Conv_block_bup(in_channels=64, out_channels=32, kernel_size = 3, stride =1, padding = 1)
        # self._conv6_1 = Conv_block_bup(in_channels=128, out_channels=32, kernel_size = 3, stride =1, padding = 1)
        # self._conv7_1 = Conv_block_bup(in_channels=256, out_channels=32, kernel_size = 3, stride =1, padding = 1)
        # self._conv5_2 = Conv_block_bup(in_channels=32, out_channels=32, kernel_size = 3, stride =2, padding = 1)

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
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #### buttom up ####
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        # #### buttom down ####
        # x5_1 = self._conv1_1(x5)
        # x6 = torch.add(x5_1, x4)
        # x6_1 = self._conv2_1(x6)
        # x7 = torch.add(x6_1, x3)
        # x7_1 = torch.nn.functional.leaky_relu(self._deconv1(x7), negative_slope=0.2)
        # x7_1= self._conv3_1(x7_1)
        # x8 = torch.add(x7_1, x2)
        # x8_1 = self._conv4_1(x8)
        # x9 = torch.add(x8_1, x1)

        # #### bup ####
        # x10 = torch.add(x9, x8_1)
        # x7_2 = self._conv5_1(x7_1)
        # x11 = self._conv5_2(torch.add(x10, x7_2))
        # x6_2 = self._conv6_1(x6_1)
        # x12 = torch.add(x11, x6_2)
        # x5_2 = self._conv7_1(x5_1)
        # x13 = torch.add(x12, x5_2)

        return x5


def resnet45():
    return ResNet(BasicBlock, [3, 4, 6, 6, 3])
