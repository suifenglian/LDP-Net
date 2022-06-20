#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Resize
import numpy as np
import cv2
import gdal
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".mat", ".tif"])


def load_img(filepath):
    img = np.array(gdal.Open(filepath).ReadAsArray(), dtype=np.double)
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = image_dir
        self.MS_LR_image_dir = os.path.join(image_dir, 'MS_LR')
        self.MS_LR_UP_image_dir = os.path.join(image_dir, 'MS_LR_UP')
        self.PAN_image_dir = os.path.join(image_dir, 'PAN')
        self.MS_image_dir = os.path.join(image_dir, 'MS')
        self.image_filenames = sorted(
            [x for x in os.listdir(self.MS_LR_image_dir) if is_image_file(x)])
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_lr = load_img(os.path.join(self.MS_LR_image_dir, self.image_filenames[index]))
        input_lr_up = load_img(os.path.join(self.MS_LR_UP_image_dir, self.image_filenames[index]))
        input_pan = load_img(os.path.join(self.PAN_image_dir, self.image_filenames[index]))

        input_lr = input_lr / 65535.
        input_pan = input_pan / 65535.
        input_lr_up = input_lr_up / 65535.

        input_pan = np.array(input_pan[None, :, :], dtype=np.float32).transpose(1, 2, 0)
        input_lr = np.array(input_lr, dtype=np.float32).transpose(1, 2, 0)
        input_lr_up = np.array(input_lr_up, dtype=np.float32).transpose(1, 2, 0)

        filename = self.image_filenames[index]
        if os.path.exists(self.MS_image_dir):
            target = load_img(os.path.join(self.MS_image_dir, self.image_filenames[index]))
            target = target / 65535.
            target = np.array(target, dtype=np.float32).transpose(1, 2, 0)
        else:
            target = np.zeros(input_lr_up.shape, dtype=np.float32)

        if self.input_transform:
            input_pan = self.input_transform(input_pan)
            input_lr = self.input_transform(input_lr)
            input_lr_up = self.input_transform(input_lr_up)
        if self.target_transform:
            target = self.target_transform(target)

        return input_lr, input_pan, input_lr_up, target, filename

    def __len__(self):
        return len(self.image_filenames)


def input_transform():
    return Compose([ToTensor()])


def target_transform():
    return Compose([ToTensor()])


def get_dataset(image_dir=None):
    return DatasetFromFolder(image_dir, input_transform=input_transform(), target_transform=target_transform())
