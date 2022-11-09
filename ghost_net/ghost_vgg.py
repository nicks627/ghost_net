import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn import random_projection

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def convert(gamma):
    g = gamma.to(torch.device('cpu'))
    o = g.detach().numpy().copy()
    return o

def random_projection(x):
    x = convert(x)
    x = torch.from_numpy(x.astype(np.float32)).clone()
    x = x.to(torch.device('cuda:0'))
    return x

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor/2)//divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)

class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # self.cheap_operation = nn.Sequential(
        #     nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
        #     nn.BatchNorm2d(new_channels),
        #     nn.ReLU(inplace=True) if relu else nn.Sequential(),
        # )
        



    def forward(self, x):
        x1 = self.primary_conv(x)
        # x2 = self.cheap_operation(x1)
        x2 = random_projection(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup,:,:]

class GhostBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=(dw_kernel_size-1)//2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None
        
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
    
    
    def forward(self, x):
        residual = x

        x = self.ghost1(x)

        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        
        if self.se is not None:
            x = self.se(x)
        
        x = self.ghost2(x)
        x += self.shortcut(residual)

        return x

class VGG_GHOST(nn.Module):
    def __init__(self, vgg_name, num_classes=10, width=1.0, dropout=0.2):
        super(VGG_GHOST, self).__init__()
        self.dropout = dropout
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(373248, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                layers += [GhostModule(in_channels, x, kernel_size=3),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


