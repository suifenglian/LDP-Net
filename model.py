#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import *


class extractor(nn.Module):
    def __init__(self, in_channel, out_channel=64):
        super(extractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel // 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(out_channel // 4 + in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(out_channel // 2 + out_channel // 4, out_channel // 2, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(out_channel, out_channel, kernel_size=2, stride=2)
        self.relu4 = nn.PReLU()

    def forward(self, x):
        x_1 = self.relu1(self.conv1(x))
        x_2 = self.relu2(self.conv2(torch.cat([x_1, x], dim=1)))
        x_3 = self.relu3(self.conv3(torch.cat([x_2, x_1], dim=1)))

        out = self.relu4(self.conv4(torch.cat([x_3, x_2], dim=1)))

        return out, x_3


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
        self.in_channel = in_channel
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
        out = stack(out, r=self.in_channel)

        return out


class Rec(nn.Module):
    def __init__(self, in_channel=4, out_channel=4):
        super(Rec, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.final_conv(x)
        x = self.relu(x)

        return x


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        self.kernel_size = 5
        self.kernel = torch.rand(1, 1, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(data=self.kernel, requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x3 = x[:, 2, :, :]
        x4 = x[:, 3, :, :]
        pad_size = self.kernel.size()[3]//2
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=pad_size, stride=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=pad_size, stride=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=pad_size, stride=1)
        x4 = F.conv2d(x4.unsqueeze(1), self.weight, padding=pad_size, stride=1)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.relu(x)

        return x


class LDP_Net(nn.Module):
    def __init__(self, in_channel=4, mid_channel=16):
        super(LDP_Net, self).__init__()
        self.extractor1 = extractor(in_channel, mid_channel)
        self.extractor2 = extractor(in_channel, mid_channel)
        self.content = Dense_encoder_decoder(2 * mid_channel, out_channel=mid_channel//2)
        self.gray = Gray(in_channel=in_channel)
        self.reblur = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=21, stride=1, padding=10)
        )
        self.rec = Rec(3 * mid_channel // 2, out_channel=in_channel)

        initialize_weights(self.extractor1, self.extractor2, self.content, self.gray, self.reblur, self.rec)

    def forward(self, x, y):
        # fuse upsampled LRMS and PAN image
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
        # gray upsampled LRMS image
        lrms_up_gray = self.gray(x)

        return ms, ms_reblur, ms_gray, pan_reblur, lrms_up_gray

