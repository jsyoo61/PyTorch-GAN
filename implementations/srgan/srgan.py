"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

from tools.tools import tdict, Timer, append, AverageMeter
from utils import *

# %%
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

opt = tdict()
opt.n_epochs=200
opt.dataset_name='img_align_celeba'
opt.batch_size=4
opt.batch_m=1
opt.lr=0.0002
opt.b1=0.5
opt.b2=0.999
opt.decay_epoch=100
opt.n_cpu=8
opt.hr_height=256
opt.hr_width=256
opt.channels=3
opt.sample_interval=100
opt.checkpoint_name='original'
print(opt)

# %%
cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()
print('number of parameters: %s'%sum(p.numel() for p in generator.parameters()))
print('number of parameters: %s'%sum(p.numel() for p in discriminator.parameters()))

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

global_timer = Timer()
epoch_timer = Timer()
iter_timer = Timer()
iter_time_meter = AverageMeter()

# %% Training
global_timer.start()
for epoch in range(opt.n_epochs):
    epoch_timer.start()
    for i, imgs in enumerate(dataloader):
        if i % opt.batch_m == 0:
            iter_timer.start()
        # Configure model input
        imgs_lr = imgs["lr"].type(Tensor)
        imgs_hr = imgs["hr"].type(Tensor)

        # Adversarial ground truths
        valid = torch.ones((imgs_lr.size(0), *discriminator_output_shape)).type(Tensor)
        fake = torch.zeros((imgs_lr.size(0), *discriminator_output_shape)).type(Tensor)

        # ------------------
        #  Train Generators
        # ------------------

        if i % opt.batch_m == 0:
            print('zero_grad_G')
            optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G = loss_G / opt.batch_m
        loss_G.backward()
        if (i+1) % opt.batch_m == 0:
            print('step_G')
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        if i % opt.batch_m == 0:
            print('zero_grad_D')
            optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D = loss_D / opt.batch_m
        loss_D.backward()
        if (i+1) % opt.batch_m == 0:
            print('step_D')
            optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        if (i+1) % opt.batch_m == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item()*opt.batch_m, loss_G.item()*opt.batch_m)
            )
            iter_time_meter.update(iter_timer.stop())
            print('time for iteration: %s (%s)'%(iter_time_meter.val, iter_time_meter.avg))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            imgs_hr_raw = imgs['hr_raw'].type(Tensor)
            with torch.no_grad():
                print('[psnr] (imgs_lr):%s, (gen_hr):%s'%(psnr(minmaxscaler(imgs_lr), imgs_hr_raw, max_val=1).mean().item(), psnr(minmaxscaler(gen_hr), imgs_hr_raw, max_val=1).mean().item()))

            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_hr_raw, imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    print('Elapsed_time for epoch(%s): %s'%epoch_timer.stop())

elapsed_time = global_timer.stop()
print('Elapsed_time for training: %s'%str(elapsed_time))
append(str(elapsed_time), 'elapsed_time.txt')
print('Average time per iteration: %s'%str(iter_time_meter.avg))
torch.save(generator.state_dict(), "saved_models/generator_%s.pth" % opt.checkpoint_name)
torch.save(discriminator.state_dict(), "saved_models/discriminator_%s.pth" % opt.checkpoint_name)

'''
batch_size 32, batch_m=4 -> 128 batch
lr = 0.0002
checkpoint_name = original

time per iter: 3.997

                 PSNR       SSIM
 generator    22.8740     0.7806
   low res    26.2965     0.7801

                 PSNR       SSIM
generator    20.1283     0.6818
  low res    24.3709     0.7694

# PSNR 20.1283, SSIM 0.6818
# 5390
# 5368
# 5552.96 /60 /60
# 5552.96 sec

'''
