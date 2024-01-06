import torch
from torch import nn

import math

import torch
from torch import nn


def random_init(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class SFE_1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,):
        super(SFE_1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=2, bias=False, stride=2)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2_1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv2_2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                                 padding=1, bias=False)
        self.conv2_3 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, dilation=2,
                                 padding=2, bias=False)
        self.conv2_4 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, dilation=3,
                                 padding=3, bias=False)
        self.bn2_1 = nn.BatchNorm2d(inplanes)
        self.bn2_2 = nn.BatchNorm2d(inplanes)
        self.bn2_3 = nn.BatchNorm2d(inplanes)
        self.bn2_4 = nn.BatchNorm2d(inplanes)
        self.conv3 = nn.Conv2d(inplanes*4, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=2, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.bn4.apply(weights_init_kaiming)
        self.bn1.apply(weights_init_kaiming)
        self.bn2_1.apply(weights_init_kaiming)
        self.bn2_2.apply(weights_init_kaiming)
        self.bn2_3.apply(weights_init_kaiming)
        self.bn2_4.apply(weights_init_kaiming)
        self.bn3.apply(weights_init_kaiming)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.conv2_1(out)
        out1 = self.bn2_1(out1)
        out1 = self.relu(out1)
        out2 = self.conv2_2(out)
        out2 = self.bn2_2(out2)
        out2 = self.relu(out2)
        out3 = self.conv2_3(out)
        out3 = self.bn2_3(out3)
        out3 = self.relu(out3)
        out4 = self.conv2_4(out)
        out4 = self.bn2_4(out4)
        out4 = self.relu(out4)
        out = torch.cat((out1, out2, out3, out4), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.conv4(residual)
        residual = self.bn4(residual)

        out += residual
        out = self.relu(out)

        return out


class SFE_2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,):
        super(SFE_2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes//2, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(inplanes//2)
        self.conv2_1 = nn.Conv2d(inplanes//2, inplanes//2, kernel_size=1, stride=1,
                                 padding=0, bias=False)
        self.conv2_2 = nn.Conv2d(inplanes//2, inplanes//2, kernel_size=3, stride=1,
                                 padding=1, bias=False)
        self.conv2_3 = nn.Conv2d(inplanes//2, inplanes//2, kernel_size=3, stride=1, dilation=2,
                                 padding=2, bias=False)
        self.conv2_4 = nn.Conv2d(inplanes//2, inplanes//2, kernel_size=3, stride=1, dilation=3,
                                 padding=3, bias=False)
        self.bn2_1 = nn.BatchNorm2d(inplanes//2)
        self.bn2_2 = nn.BatchNorm2d(inplanes//2)
        self.bn2_3 = nn.BatchNorm2d(inplanes//2)
        self.bn2_4 = nn.BatchNorm2d(inplanes//2)
        self.conv3 = nn.Conv2d(inplanes*2, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.bn4.apply(weights_init_kaiming)
        self.bn1.apply(weights_init_kaiming)
        self.bn2_1.apply(weights_init_kaiming)
        self.bn2_2.apply(weights_init_kaiming)
        self.bn2_3.apply(weights_init_kaiming)
        self.bn2_4.apply(weights_init_kaiming)
        self.bn3.apply(weights_init_kaiming)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.conv2_1(out)
        out1 = self.bn2_1(out1)
        out1 = self.relu(out1)
        out2 = self.conv2_2(out)
        out2 = self.bn2_2(out2)
        out2 = self.relu(out2)
        out3 = self.conv2_3(out)
        out3 = self.bn2_3(out3)
        out3 = self.relu(out3)
        out4 = self.conv2_4(out)
        out4 = self.bn2_4(out4)
        out4 = self.relu(out4)
        out = torch.cat((out1, out2, out3, out4), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.conv4(residual)
        residual = self.bn4(residual)

        out += residual
        out = self.relu(out)

        return out

class SFE_3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,):
        super(SFE_3, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(inplanes // 4)
        self.conv2_1 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=1, stride=1,
                                 padding=0, bias=False)
        self.conv2_2 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, stride=1,
                                 padding=1, bias=False)
        self.conv2_3 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, stride=1, dilation=2,
                                 padding=2, bias=False)
        self.conv2_4 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, stride=1, dilation=3,
                                 padding=3, bias=False)
        self.bn2_1 = nn.BatchNorm2d(inplanes // 4)
        self.bn2_2 = nn.BatchNorm2d(inplanes // 4)
        self.bn2_3 = nn.BatchNorm2d(inplanes // 4)
        self.bn2_4 = nn.BatchNorm2d(inplanes // 4)
        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.bn4.apply(weights_init_kaiming)
        self.bn1.apply(weights_init_kaiming)
        self.bn2_1.apply(weights_init_kaiming)
        self.bn2_2.apply(weights_init_kaiming)
        self.bn2_3.apply(weights_init_kaiming)
        self.bn2_4.apply(weights_init_kaiming)
        self.bn3.apply(weights_init_kaiming)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.conv2_1(out)
        out1 = self.bn2_1(out1)
        out1 = self.relu(out1)
        out2 = self.conv2_2(out)
        out2 = self.bn2_2(out2)
        out2 = self.relu(out2)
        out3 = self.conv2_3(out)
        out3 = self.bn2_3(out3)
        out3 = self.relu(out3)
        out4 = self.conv2_4(out)
        out4 = self.bn2_4(out4)
        out4 = self.relu(out4)
        out = torch.cat((out1, out2, out3, out4), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.conv4(residual)
        residual = self.bn4(residual)

        out += residual
        out = self.relu(out)

        return out


