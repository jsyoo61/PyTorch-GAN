import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def minmaxscaler(x):
    x_min = x.min()
    x_max = x.max()

    return (x-x_min)/(x_max-x_min)

def psnr(a, b, max_val):
    '''
    Compute Peak Signal-to-Noise Ratio
    a: torch.tensor
    b: torch.tensor
    max_val: maximum value of image

    returns: single value
    '''
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    mse = torch.mean(((a-b)**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse)
    return v

def psnr_shift(a,b, max_val, shift=1, shift_dir=''):
    '''
    Compute Peak Signal-to-Noise Ratio, mean value of PSNR shifting in 4~8 directions
    a: torch.tensor
    b: torch.tensor
    max_val: maximum value of image
    shift: number of pixels to shift

    returns: mean of psnr
    '''
    # ----------
    # directions
    #
    # ----------
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    # 1. original
    mse = torch.mean(((a-b)**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v_o = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse)

    mse_r = torch.mean(((a[...,:,shift:]-b[...,:,:-shift])**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v_r = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse_r)

    mse_l = torch.mean(((a[...,:,:-shift]-b[...,:,shift:])**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v_l = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse_l)

    mse_u = torch.mean(((a[...,:-shift,:]-b[...,shift:,:])**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v_u = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse_u)

    mse_d = torch.mean(((a[...,shift:,:]-b[...,:-shift,:])**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v_d = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse_d)

    v = torch.mean(torch.stack([v_o,v_r,v_l,v_u,v_d],dim=0),dim=0)
    # v_ = torch.stack([v_o,v_r,v_l,v_u,v_d],dim=0)
    return v


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def filter_state_dict(checkpoint_name):
    state_dict = torch.load('saved_models/%s.pth'%checkpoint_name)
    state_dict_ = {}
    for key, value in state_dict.items():
        key_ = '.'.join(key.split('.')[1:])
        print(key_)
        state_dict_[key_] = value

    torch.save(state_dict_, 'saved_models/%s_.pth'%checkpoint_name)

# x=[torch.arange(i,i+10) for i in range(10)]
# torch.mean(*x)
# sum(x)
#     a_r = a[:, shift:]
#     b_r = b[:, :-shift]
#     x=torch.arange(30).reshape(1,2,3,5)
#     x=torch.arange(30).reshape(6,5)
#     x[...,:,2:].shape
#
#
#     return torch.mean()
#
