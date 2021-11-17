#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import torch.utils.data as data
import scipy.io as scio
from torchvision.transforms import Compose, ToTensor
import numpy as np
import cv2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".mat"])


def load_img(filepath):
    y = scio.loadmat(filepath)
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None, Full=False, Test=False):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = image_dir
        self.LRMS_UP_image_dir = os.path.join(image_dir, 'MS_LR_UP')
        self.LRMS_image_dir = os.path.join(image_dir, 'MS_LR')
        self.PAN_image_dir = os.path.join(image_dir, 'PAN_LR')
        self.HRMS_image_dir = os.path.join(image_dir, 'MS')
        self.image_filenames = sorted(
            [x for x in os.listdir(self.LRMS_UP_image_dir) if is_image_file(x)])
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.test = Test
        self.full_resolution = Full

    def __getitem__(self, index):
        if self.full_resolution:
            input_lr = load_img(os.path.join(self.LRMS_image_dir, self.image_filenames[index])).get(
                'MS_LR')
            input_lr_up = load_img(os.path.join(self.LRMS_UP_image_dir, self.image_filenames[index])).get(
                'MS_LR_UP')
            input_pan = load_img(
                os.path.join(self.PAN_image_dir, self.image_filenames[index])).get('PAN_LR')
            filename = self.image_filenames[index]

            input_lr = np.array(input_lr, dtype=np.float32)
            input_lr_up = np.array(input_lr_up, dtype=np.float32)
            input_pan = np.array(input_pan[:, :, None], dtype=np.float32)

            if self.input_transform:
                input_lr = self.input_transform(input_lr)
                input_lr_up = self.input_transform(input_lr_up)
                input_pan = self.input_transform(input_pan)

            return input_pan, input_lr_up, filename

        if self.test:
            input_lr = load_img(os.path.join(self.LRMS_image_dir, self.image_filenames[index])).get(
                'MS_LR')
            input_lr_up = load_img(os.path.join(self.LRMS_UP_image_dir, self.image_filenames[index])).get('MS_LR_UP')
            input_pan = load_img(
                os.path.join(self.PAN_image_dir, self.image_filenames[index])).get('PAN_LR')
            target = load_img(os.path.join(self.HRMS_image_dir, self.image_filenames[index])).get('MS')
        else:
            input_lr = load_img(os.path.join(self.LRMS_image_dir, self.image_filenames[index])).get(
                'MS_LR_patch')
            input_lr_up = load_img(os.path.join(self.LRMS_UP_image_dir, self.image_filenames[index])).get('MS_LR_UP_patch')
            input_pan = load_img(os.path.join(self.PAN_image_dir, self.image_filenames[index])).get('PAN_LR_patch')
            target = load_img(os.path.join(self.HRMS_image_dir, self.image_filenames[index])).get('MS_patch')

        filename = self.image_filenames[index]

        input_lr = np.array(input_lr, dtype=np.float32)
        input_lr_up = np.array(input_lr_up, dtype=np.float32)
        input_pan = np.array(input_pan[:, :, None], dtype=np.float32)
        target = np.array(target, dtype=np.float32)

        if self.input_transform:
            input_lr = self.input_transform(input_lr)
            input_lr_up = self.input_transform(input_lr_up)
            input_pan = self.input_transform(input_pan)

        if self.target_transform:
            target = self.target_transform(target)

        return input_lr, input_pan, input_lr_up, target, filename

    def __len__(self):
        return len(self.image_filenames)


class Stretch:
    def __call__(self, pic):
        return linear(pic)


def linear(pic):
    return pic * 2.0 - 1.0


def input_transform():
    return Compose([ToTensor(), Stretch()])


def target_transform():
    return Compose([ToTensor(), Stretch()])


def get_dataset(image_dir=None, full_flag=False, test_flag=False):
    return DatasetFromFolder(image_dir, input_transform=input_transform(), target_transform=target_transform(),
                             Full=full_flag, Test=test_flag)
