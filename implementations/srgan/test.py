# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.utils import save_image, make_grid

from datasets import *
from models import *

from utils import *
import os
import pdb
import argparse

os.makedirs("images_test", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--sample_interval", type=int, default=200, help="interval between saving image samples")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--checkpoint_name", type=str, default='default', help="name of checkpoint")
parser.add_argument("--grayscale", type=eval, default='True', help="grayscale or RGB")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

os.makedirs('test', exist_ok=True)
generator = GeneratorResNet()
generator.cuda()
generator.load_state_dict(torch.load('saved_models/generator_%s.pth' % opt.checkpoint_name))

dataloader = DataLoader(ImageDataset('../../data/img_align_celeba_eval', hr_shape=(256,256)),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=8,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

psnr_gen_list = []
psnr_lr_list = []
ssim_gen_list = []
ssim_lr_list = []

print('Grayscale: %s'%opt.grayscale)
for i, imgs in enumerate(dataloader):
    with torch.no_grad():

        imgs_lr = imgs['lr'].type(Tensor)
        imgs_hr = imgs['hr'].type(Tensor)
        imgs_hr_raw = imgs['hr_raw'].type(Tensor)

        gen_hr = generator(imgs_lr)

        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        imgs_lr = minmaxscaler(imgs_lr)
        imgs_hr = minmaxscaler(imgs_hr)
        gen_hr = minmaxscaler(gen_hr)

        if opt.grayscale:
            gen_hr_gray = gen_hr.mean(dim=1)[:,None,:,:]
            imgs_lr_gray = imgs_lr.mean(dim=1)[:,None,:,:]
            imgs_hr_raw_gray = imgs_hr_raw.mean(dim=1)[:,None,:,:]

            psnr_gen = psnr(gen_hr_gray, imgs_hr_raw_gray, max_val=1)
            psnr_lr = psnr(imgs_lr_gray, imgs_hr_raw_gray, max_val=1)
            ssim_gen = ssim(gen_hr_gray, imgs_hr_raw_gray, size_average=False)
            ssim_lr = ssim(imgs_lr_gray, imgs_hr_raw_gray, size_average=False)
        else:
            psnr_gen = psnr(gen_hr, imgs_hr_raw, max_val=1)
            psnr_lr = psnr(imgs_lr, imgs_hr_raw, max_val=1)
            ssim_gen = ssim(gen_hr, imgs_hr_raw, size_average=False)
            ssim_lr = ssim(imgs_lr, imgs_hr_raw, size_average=False)

        psnr_gen_list.append(psnr_gen)
        psnr_lr_list.append(psnr_lr)
        ssim_gen_list.append(ssim_gen)
        ssim_lr_list.append(ssim_lr)

        batches_done = i
        if batches_done % opt.sample_interval == 0:
            print('%10s %10s %10s'%('','PSNR','SSIM'))
            print('%10s %10.4f %10.4f'%('generator',psnr_gen.mean().item(),ssim_gen.mean().item()))
            print('%10s %10.4f %10.4f'%('low res',psnr_lr.mean().item(),ssim_lr.mean().item()))

            # Save image grid with upsampled inputs and SRGAN outputs
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            imgs_hr_raw = make_grid(imgs_hr_raw, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_hr_raw, imgs_lr, gen_hr), -1)
            save_image(img_grid, "images_test/%d.png" % batches_done, normalize=False)

psnr_gen = torch.cat(psnr_gen_list).mean().item()
psnr_lr = torch.cat(psnr_lr_list).mean().item()
ssim_gen = torch.cat(ssim_gen_list).mean().item()
ssim_lr = torch.cat(ssim_lr_list).mean().item()
print('-'*32)
print('Grayscale: %s'%opt.grayscale)
print('%10s %10s %10s'%('','PSNR','SSIM'))
print('%10s %10.4f %10.4f'%('generator',psnr_gen,ssim_gen))
print('%10s %10.4f %10.4f'%('low res',psnr_lr,ssim_lr))

#
# imgs = next(iter(dataloader))
# imgs['lr'].shape
# imgs['hr'].max()
# imgs['hr'].min()
# imgs['hr_raw'].max()
# imgs['hr_raw'].min()
# imgs_lr = imgs['lr'].type(Tensor)
# imgs_hr = imgs['hr'].type(Tensor)
# imgs_hr_raw = imgs['hr_raw']
#
# # %%
# with torch.no_grad():
#     gen_hr = generator(imgs_lr).detach().cpu()
#
#     gen_hr = minmaxscaler(gen_hr)
#     imgs_lr = F.interpolate(minmaxscaler(imgs_lr), scale_factor=4).detach().cpu()
#     gen_hr_gray = gen_hr.mean(dim=1)
#     imgs_lr_gray = imgs_lr.mean(dim=1)
#     imgs_hr_raw_gray = imgs_hr_raw.mean(dim=1)
#     print(psnr(gen_hr, imgs_hr_raw, max_val=1))
#     print(psnr(imgs_lr, imgs_hr_raw, max_val=1))
#     print(psnr(gen_hr.mean(dim=1), imgs_hr_raw.mean(dim=1), max_val=1))
#     print(psnr(imgs_lr.mean(dim=1), imgs_hr_raw.mean(dim=1), max_val=1))
#     print(psnr_shift(gen_hr, imgs_hr_raw, max_val=1,shift=3))
#     print(psnr_shift(imgs_lr, imgs_hr_raw, max_val=1,shift=3))
#     print(psnr_shift(gen_hr.mean(dim=1), imgs_hr_raw.mean(dim=1), max_val=1, shift=4))
#     print(psnr_shift(imgs_lr.mean(dim=1), imgs_hr_raw.mean(dim=1), max_val=1, shift=4))
#     ssim(gen_hr, imgs_hr_raw, size_average=False)
#     ssim(imgs_lr, imgs_hr_raw, size_average=False)
#     ssim(gen_hr_gray[:,None,:,:], imgs_hr_raw_gray[:,None,:,:], size_average=False)
#     ssim(imgs_lr_gray[:,None,:,:], imgs_hr_raw_gray[:,None,:,:], size_average=False)
# (gen_hr_gray - imgs_hr_raw_gray).mean()
# (imgs_lr_gray - imgs_hr_raw_gray).mean()
# with torch.no_grad():
#     gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
#     imgs_hr_raw = make_grid(imgs_hr_raw, nrow=1, normalize=True)
#     imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
#     img_grid = torch.cat((imgs_hr_raw, gen_hr, imgs_lr), -1)
#     save_image(img_grid,'test/1.png', normalize=False)
#
# grayscale = transforms.Grayscale()
# totensor = transforms.ToTensor()
# toPIL = transforms.ToPILImage()
# toPIL(gen_hr_gray)
#
# imgs_hr = minmaxscaler(imgs_hr)
# torch.isclose(imgs_hr, imgs_hr_raw)
# imgs_hr_raw.min()
# imgs_hr_raw.max()
# imgs_hr.min()
# imgs_hr.max()
#
# pdb.set_trace()
#
# gen_hr[0].cpu().shape
# torch.isclose(totensor(grayscale(toPIL(gen_hr[0].cpu()))), gen_hr[0].cpu().mean(dim=0))
# toPIL(gen_hr[0].cpu().mean(dim=0))
# psnr(gen_hr[0].cpu().mean(dim=0), imgs_hr_raw[0].cpu().mean(dim=0), max_val=1)
#
# psnr(totensor(grayscale(toPIL(gen_hr[0].cpu()))), totensor(grayscale(toPIL(imgs_hr_raw[0].cpu()))), max_val=1)
# grayscale(toPIL(imgs_lr[0].cpu()))
# grayscale(toPIL(gen_hr[0].cpu()))
# grayscale(toPIL(imgs_hr_raw[0].cpu()))
# transforms.Grayscale()(imgs_lr_PIL)
# transforms.ToTensor()()
# imgs_lr_PIL = transforms.ToPILImage()(imgs_lr[0].cpu())
# gen_hr_PIL = transforms.ToPILImage()(gen_hr[0].cpu())
# imgs_hr_raw_PIL = transforms.ToPILImage()(imgs_hr_raw[0].cpu())
# imgs_lr[0].min(), imgs_lr[0].max(), imgs_lr.min(), imgs_lr.max()
# gen_hr[0].min(), gen_hr[0].max(), gen_hr.min(), gen_hr.max()
# imgs_hr_raw[0].min(), imgs_hr_raw[0].max(), imgs_hr_raw.min(), imgs_hr_raw.max()
# x
# a=gen_hr
# b=imgs_hr_raw
# max_val=1
# shift=1
#
# from PIL import Image
# img=Image.open('../../data/img_align_celeba_eval/162771.jpg')
# img
# type(img)
