#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import visdom
import math
import torch.nn.functional as F


class extractor(nn.Module):
    def __init__(self, in_channel, out_channel=64):
        super(extractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 4, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel // 4 + in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel // 2 + out_channel // 4, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=2, stride=2),
            nn.PReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x1, x], dim=1))
        x3 = self.conv3(torch.cat([x2, x1], dim=1))
        out = self.conv4(torch.cat([x3, x2], dim=1))

        return out, x3


class Dense_encoder_decoder(nn.Module):
    def __init__(self, in_channel, out_channel=64):
        super(Dense_encoder_decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3 * in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
            nn.PReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], dim=1))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
        out = self.up(x4)

        return out


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()


class Gray(nn.Module):
    def __init__(self, in_channel=4, retio=4):
        super(Gray, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel * retio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel * retio, in_channel, bias=False),
            nn.Sigmoid(),
            nn.Softmax()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        y = self.avg(x2).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        out = torch.sum(out, dim=1, keepdim=True)
        out = torch.cat([out, out, out, out], dim=1)

        return out


class Rec(nn.Module):
    def __init__(self, in_channel=4, out_channel=4):
        super(Rec, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.final_conv(x)
        x = self.relu(x)

        return x


class LDP_Net(nn.Module):
    def __init__(self, in_channel=4, mid_channel=64):
        super(LDP_Net, self).__init__()
        self.extractor1 = extractor(in_channel, mid_channel)
        self.extractor2 = extractor(in_channel, mid_channel)
        self.content = Dense_encoder_decoder(2 * mid_channel, out_channel=mid_channel//2)
        self.gray = Gray()
        self.reblur = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=41, stride=1, padding=20),
            nn.Tanh()
        )
        self.rec = Rec(3 * mid_channel // 2, out_channel=in_channel)

        initialize_weights(self.extractor1, self.extractor2, self.content, self.gray, self.reblur, self.rec)

    def forward(self, x, y):
        # fuse LRMS and PAN image
        x0, x_3 = self.extractor1(x)
        y0, y_3 = self.extractor2(y)
        content_out = self.content(torch.cat([x0, y0], dim=1))
        content_out = torch.cat([content_out, x_3, y_3], dim=1)
        ms = self.rec(content_out)

        # gray MS image
        ms_gray = self.gray(ms)
        # reblur MS image
        ms_reblur = self.reblur(ms)

        # reblur PAN image
        pan_reblur = self.reblur(y)
        # gray LRMS_image
        lrms_up_gray = self.gray(x)

        return ms, ms_reblur, ms_gray, pan_reblur, lrms_up_gray


if __name__ == '__main__':
    net = LDP_Net()
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters : %.3f M' % (num_params / 1e6))
    print(net)