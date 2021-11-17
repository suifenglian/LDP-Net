#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import os
import numpy as np
import random
from model import LDP_Net
from dataset import get_dataset
from vis_tools import Visualizer
import torch.backends.cudnn as cudnn
from eval_matrics import sam, sCC
import scipy.io
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Pytorch LDP_Net')
# model
parser.add_argument('--model', type=str, default='LDP_Net')
# dataset
parser.add_argument('--dataset', type=str, default='WV2')

# train
parser.add_argument('--pixel_loss_type', type=str, default='L2')
parser.add_argument('--in_nc', type=int, default=4, help='number of input image channels')
parser.add_argument('--mid_nc', type=int, default=128, help='number of middle feature maps')
parser.add_argument('--out_nc', type=int, default=4, help='number of output image channels')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
parser.add_argument('--resume', type=str, default='', help='path to model checkpoint')
parser.add_argument('--start_epoch', type=int, default=1, help='restart epoch number for training')
parser.add_argument('--threads', type=int, default=0, help='number of threads')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--step', type=int, default=5,
                    help='Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=5')
parser.add_argument('--train_dir', type=str, default='',
                    help='image path to training data directory')
parser.add_argument('--log', type=str, default='log/')

# test
parser.add_argument('--test', type=bool, default=False, help='test')
parser.add_argument('--test_full', type=bool, default=False, help='test full-resolution')
parser.add_argument('--test_dir', type=str, default='', help='image path to testing data directory')
parser.add_argument('--savedir', type=str, default='')
parser.add_argument('--pretrained', type=str, default='', help='path to model parameters')
parser.add_argument('--test_batch_size', type=int, default=2, help='testing batch size')
opt = parser.parse_args()
print(opt)

seed = random.randint(1, 10000)
print("Random Seed: ", seed)
torch.manual_seed(seed)
if opt.cuda:
    torch.cuda.manual_seed(seed)
cudnn.benchmark = True

# build network
print('==>building network...')
model = LDP_Net(in_channel=opt.in_nc, mid_channel=opt.mid_nc)

# pixel loss
pixel_loss = torch.nn.L1Loss()
if opt.pixel_loss_type == 'L1':
    pixel_loss = torch.nn.L1Loss()
elif opt.pixel_loss_type == 'L2':
    pixel_loss = torch.nn.MSELoss()
elif opt.pixel_loss_type == 'SmoothL1':
    pixel_loss = torch.nn.SmoothL1Loss()

# KL loss
kl_loss = torch.nn.KLDivLoss(reduction='sum')

# set GPU
if opt.cuda and not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')
print("===> Setting GPU")
if opt.cuda:
    print('cuda_mode:', opt.cuda)
    model = model.cuda()
    pixel_loss = pixel_loss.cuda()
    kl_loss = kl_loss.cuda()

# optimizer
print("===> Setting Optimizer")
optim = torch.optim.Adam(model.parameters(), lr=opt.lr)

# visualizer
train_vis = Visualizer(env='training')


# training
def train(train_dataloader, model, optim, kl_loss, pixel_loss):
    print('==>Training...')
    for epoch in range(opt.start_epoch, opt.num_epochs + 1):
        train_process(train_dataloader, model, optim, kl_loss, pixel_loss,
                      epoch, epochs=opt.num_epochs)
        if epoch % 2 == 0:
            save_checkpoint(model, epoch)


# testing
def test(test_dataloader, model, save_img_dir):
    print('==>Testing...')
    if opt.test_full:
        test_process_full(test_dataloader, model, save_img_dir)
    else:
        test_process(test_dataloader, model, save_img_dir)


# train every epoch
def train_process(dataloader, model, optim, kl_loss, pixel_loss, epoch=1,
                  epochs=10):
    lr = adjust_learning_rate(epoch - 1)
    for param_group in optim.param_groups:
        param_group["lr"] = lr
        print("epoch =", epoch, "lr =", optim.param_groups[0]["lr"])

    for iteration, batch in enumerate(dataloader):
        input_lr, input_pan, input_lr_up, target = batch[0], batch[1], batch[2], batch[3]
        model.train()
        if opt.cuda:
            input_lr = input_lr.cuda()
            input_pan = input_pan.cuda()
            input_lr_up = input_lr_up.cuda()
            target = target.cuda()
        # -----------------------------------------------
        # training model
        # ------------------------------------------------
        pan = input_pan
        pan_multi = torch.cat([pan, pan, pan, pan], dim=1)
        ms, ms_reblur, ms_gray, pan_reblur, lrms_up_gray = model(input_lr_up, pan_multi)
        optim.zero_grad()

        # spectral_loss
        ms_downsample = F.interpolate(ms, [32, 32], mode='bilinear', align_corners=True)
        loss_spectral = pixel_loss(ms_reblur, input_lr_up) + 10 * pixel_loss(ms_downsample, input_lr)
        # loss_spectral = pixel_loss(ms_reblur, input_lr_up) + 5 * pixel_loss(ms_downsample, input_lr)

        # spatial_loss
        loss_spatial = pixel_loss(pan_reblur, lrms_up_gray) + 10 * pixel_loss(ms_gray, pan_multi)

        # KL loss
        res1 = input_lr_up - lrms_up_gray
        res2 = ms - pan_multi
        loss_kl = 0.01 * kl_loss(res1.softmax(dim=-1).log(), res2.softmax(dim=-1))

        # total loss
        loss = loss_spatial + loss_spectral + loss_kl
        loss.backward()
        optim.step()

        input_lr = (input_lr + 1) / 2.0
        input_lr_up = (input_lr_up + 1) / 2.0
        pan_multi = (pan_multi + 1) / 2.0
        target = (target + 1) / 2.0

        ms = (ms + 1) / 2.0
        ms_downsample = (ms_downsample + 1) / 2.0
        ms_reblur = (ms_reblur + 1) / 2.0
        ms_gray = (ms_gray + 1) / 2.0
        lrms_up_gray = (lrms_up_gray + 1) / 2.0
        pan_reblur = (pan_reblur + 1) / 2.0

        sam_data = sam(target[0].cpu().detach().numpy(), ms[0].cpu().detach().numpy())
        scc_data = sCC(target[0].cpu().detach().numpy(), ms[0].cpu().detach().numpy())

        train_vis.plot('loss', loss.item())
        train_vis.img('MS_LR', input_lr[0:4, :3, :, :].data.mul(255).cpu().detach().numpy())
        train_vis.img('MS_LR_UP', input_lr_up[0:4, :3, :, :].data.mul(255).cpu().detach().numpy())
        train_vis.img('PAN', pan_multi[0:4, :3, :, :].data.mul(255).cpu().detach().numpy())
        train_vis.img('GT', target[0:4, :3, :, :].data.mul(255).cpu().detach().numpy())

        train_vis.img('Predict', ms[0:4, :3, :, :].data.mul(255).cpu().detach().numpy())
        train_vis.img('Predict_downsample', ms_downsample[0:4, :3, :, :].data.mul(255).cpu().detach().numpy())
        train_vis.img('Predict_reblur', ms_reblur[0:4, :3, :, :].data.mul(255).cpu().detach().numpy())
        train_vis.img('Predict_gray', ms_gray[0:4, :3, :, :].data.mul(255).cpu().detach().numpy())
        train_vis.img('lrms_up_gray', lrms_up_gray[0:4, :3, :, :].data.mul(255).cpu().detach().numpy())
        train_vis.img('pan_reblur', pan_reblur[0:4, :3, :, :].data.mul(255).cpu().detach().numpy())

        print('epoch:[{}/{}] batch:[{}/{}] loss:{:.5f} loss_spatial:{:.5f} loss_spectral:{:.5f} '
              'loss_kl: {:.5f} | sam:{:.5f} scc:{:.5f} '.format(epoch, epochs, iteration, len(dataloader), loss,
                                                                loss_spatial, loss_spectral, loss_kl,
                                                                sam_data, scc_data))


def test_process(test_dataloader, model, save_img_dir):
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    model.eval()

    for idx, batch in enumerate(test_dataloader):
        input_pan, input_lr_up, target, filename = batch[1], batch[2], batch[3], batch[4]

        if opt.cuda:
            input_pan = input_pan.cuda()
            input_lr_up = input_lr_up.cuda()
            target = target.cuda()
        pan = input_pan
        input_pan = torch.cat([pan, pan, pan, pan], dim=1)

        prediction = model(input_lr_up, input_pan)
        ms = prediction[0]

        target = (target + 1) / 2.0
        ms = (ms + 1) / 2.0

        save_GT_dir = os.path.join(save_img_dir, 'GT')
        save_pre_dir = os.path.join(save_img_dir, 'prediction')
        if not os.path.exists(save_pre_dir):
            os.mkdir(save_pre_dir)
        if not os.path.exists(save_GT_dir):
            os.mkdir(save_GT_dir)

        for i in range(opt.test_batch_size):
            out = ms[i].data.cpu().detach().numpy()
            out = np.transpose(out, (1, 2, 0))
            scipy.io.savemat(os.path.join(save_pre_dir, '%s' % (filename[i])), {'out': out})

            label = target[i].data.cpu().detach().numpy()
            label = np.transpose(label, (1, 2, 0))
            scipy.io.savemat(os.path.join(save_GT_dir, '%s' % (filename[i])), {'label': label})


def test_process_full(test_dataloader, model, save_img_dir):
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    model.eval()
    tbar = tqdm(test_dataloader, ncols=50)
    for idx, batch in enumerate(tbar):
        pan, input_lr_up, filename = batch[0], batch[1], batch[2]

        if opt.cuda:
            pan = pan.cuda()
            input_lr_up = input_lr_up.cuda()
        input_pan = torch.cat([pan, pan, pan, pan], dim=1)
        prediction = model(input_lr_up, input_pan)
        ms = prediction[0]
        ms = (ms + 1) / 2.0

        save_pre_dir = os.path.join(save_img_dir, 'prediction')
        if not os.path.exists(save_pre_dir):
            os.mkdir(save_pre_dir)

        for i in range(opt.test_batch_size):
            out = ms[i].data.cpu().detach().numpy()
            out = np.transpose(out, (1, 2, 0))
            scipy.io.savemat(os.path.join(save_pre_dir, '%s' % (filename[i])), {'out': out})


def adjust_learning_rate(epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    if lr < 1e-10:
        lr = 1e-10
    return lr


def save_checkpoint(model, epoch):
    model_folder = opt.model + "_" + opt.dataset + "_model_para/"
    model_parm_path = model_folder + "epoch{}.pkl".format(epoch)
    model_state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model_state, model_parm_path)
    print("Checkpoint saved to {}".format(model_parm_path, ))


if opt.pretrained and opt.test:
    # conduct testing procedure
    print('==>loading test data...')
    test_dataset = get_dataset(opt.test_dir, full_flag=opt.test_full, test_flag=opt.test)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=opt.test_batch_size, shuffle=True,
                                      num_workers=opt.threads)
    if os.path.isfile(opt.pretrained):
        print('==> loading model {}'.format(opt.pretrained))
        model_weights = torch.load(opt.pretrained)
        model.load_state_dict(model_weights['model'].state_dict())
        test(test_dataloader, model, opt.savedir)
    else:
        print('==> no generator model found at {}'.format(opt.pretrained_gen))
else:
    # conduct training procedure
    print('==>loading training data...')
    train_dataset = get_dataset(opt.train_dir)
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                                       num_workers=opt.threads)
    if opt.resume:
        if os.path.isfile(opt.resume):
            model_checkpoint = torch.load(opt.resume_gen)
            opt.start_epoch = model_checkpoint['epoch'] + 1
            print('==>start training at epoch {}'.format(opt.start_epoch))
            model.load_state_dict(model_checkpoint['model'].state_dict())
            print("===> resume Training...")
            train(train_dataloader, model, optim, kl_loss, pixel_loss)
        else:
            print('==> cannot start training at epoch {}'.format(opt.start_epoch))
    else:
        train(train_dataloader, model, optim, kl_loss, pixel_loss)
