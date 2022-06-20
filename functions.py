import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn


def image_clip(image, min=0, max=1):
    min = torch.Tensor([min]).float().cuda()
    max = torch.Tensor([max]).float().cuda()
    return torch.min(torch.max(image, min), max)


def stack(pan, r=4):
    return torch.cat([pan for _ in range(r)], dim=1)


class Smooth(nn.Module):
    def __init__(self, in_nc=4):
        super(Smooth, self).__init__()
        self.chennel = in_nc
        kernel = np.ones([3, 3]) * (1/9)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.chennel, axis=0)
        kernel = kernel.cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, padding=1, groups=self.chennel)
        return x
