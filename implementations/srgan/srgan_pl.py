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

import torch.nn as nn
import torch.nn.functional as F
import torch

from tools.tools import tdict, Timer, append, AverageMeter
from utils import *
from aggregation import aggregate_grad, distribute_all

import pdb

# %%
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--batch_m", type=int, default=1, help="batch multiplier. iterate over n batches and then apply gradients")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--checkpoint_name", type=str, default='default', help="name of checkpoint")

opt = parser.parse_args()
print(opt)

# %%
cuda = torch.cuda.is_available()
n_cuda = torch.cuda.device_count()
hr_shape = (opt.hr_height, opt.hr_width)
print('n_cuda: %s'%n_cuda)

# Initialize generator and discriminator
generator_list = []
discriminator_list = []
feature_extractor_list = []

optimizer_G_list = []
optimizer_D_list = []
for i in range(n_cuda):
    generator = GeneratorResNet().cuda(i)
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).cuda(i)
    feature_extractor = FeatureExtractor().cuda(i)

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    generator_list.append(generator)
    discriminator_list.append(discriminator)
    feature_extractor_list.append(feature_extractor)
    optimizer_G_list.append(optimizer_G)
    optimizer_D_list.append(optimizer_D)

optimizer_G = optimizer_G_list[0]
optimizer_D = optimizer_D_list[0]

print('number of parameters (generator): %s'%sum(p.numel() for p in generator_list[0].parameters()))
print('number of parameters (discriminator): %s'%sum(p.numel() for p in discriminator_list[0].parameters()))
for generator, discriminator, feature_extractor in zip(generator_list, discriminator_list, feature_extractor_list):
    generator_device = next(generator.parameters()).device
    discriminator_device = next(discriminator.parameters()).device
    feature_extractor_device = next(feature_extractor.parameters()).device
    print('models on device: generator(%s), discriminator(%s), feature_extractor(%s)'%(generator_device, discriminator_device, feature_extractor_device))

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=True
)

global_timer = Timer()
epoch_timer = Timer()
iter_timer = Timer()
iter_time_meter = AverageMeter()

# ----------
#  Training
# ----------
global_timer.start()
for epoch in range(opt.epoch, opt.n_epochs):
    epoch_timer.start()

    imgs_list = []
    for i, imgs in enumerate(dataloader):
        if i % n_cuda == 0:
            iter_timer.start()

        imgs_list.append(imgs)
        if len(imgs_list) < n_cuda:
            continue

        print('zero_grad_G')
        optimizer_G.zero_grad()
        print('zero_grad_D')
        optimizer_D.zero_grad()
        for imgs, generator, discriminator, feature_extractor in zip(imgs_list, generator_list, discriminator_list, feature_extractor_list):
            device = next(generator.parameters()).device
            with torch.cuda.device(device):
                # ------------------
                #  Train Generators
                # ------------------

                # Configure model input
                imgs_lr = imgs["lr"].cuda()
                imgs_hr = imgs["hr"].cuda()

                # Adversarial ground truths
                valid = torch.ones((imgs_lr.size(0), *discriminator.output_shape), device=device)
                fake = torch.zeros((imgs_lr.size(0), *discriminator.output_shape), device=device)

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

                loss_G = loss_G
                loss_G.backward()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Loss of real and fake images
                loss_real = criterion_GAN(discriminator(imgs_hr), valid)
                loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

                # Total loss
                loss_D = (loss_real + loss_fake) / 2

                loss_D = loss_D
                loss_D.backward()

        aggregate_grad(generator_list[1:], generator_list[0])
        print('step_G')
        optimizer_G.step()
        distribute_all(generator_list[0], generator_list[1:])

        aggregate_grad(discriminator_list[1:], discriminator_list[0])
        print('step_D')
        optimizer_D.step()
        distribute_all(discriminator_list[0], discriminator_list[1:])

        imgs_list = []

        # --------------
        #  Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item()*n_cuda, loss_G.item()*n_cuda)
        )
        iter_time_meter.update(iter_timer.stop())
        print('time for iteration: %s (%s)'%(iter_time_meter.val, iter_time_meter.avg))

        batches_done = epoch * len(dataloader) + i+1
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            imgs_hr_raw = imgs['hr_raw'].to(device)
            print('[psnr] (imgs_lr):%s, (gen_hr):%s'%(psnr(minmaxscaler(imgs_lr), imgs_hr_raw, max_val=1).mean(), psnr(minmaxscaler(gen_hr), imgs_hr_raw, max_val=1).mean()))

            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)
    elapsed_time = epoch_timer.stop()
    print('Elapsed_time: %s'%elapsed_time)
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
elapsed_time = global_timer.stop()
print(str(elapsed_time))
append(str(elapsed_time), 'elapsed_time.txt')
torch.save(generator.state_dict(), "saved_models/generator_%s.pth" % opt.checkpoint_name)
torch.save(discriminator.state_dict(), "saved_models/discriminator_%s.pth" % opt.checkpoint_name)

2 **((1/4)*np.log2(6))
