import os
import argparse
from models import *
from datasets import *
from tools.tools import Timer, AverageMeter
import socket
from utils import *
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

def main():

    os.makedirs("images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
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

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--master_addr', default=str(socket.gethostbyname(socket.gethostname())), type=str, help='master ip address')
    parser.add_argument('--master_port', default='8888', type=str, help='master port')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = '10.57.23.164'
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '20000'
    print(args)
    mp.spawn(train, nprocs=args.gpus, args=(args,))

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)

    hr_shape = (args.hr_height, args.hr_width)
    generator = GeneratorResNet()
    discriminator = Discriminator(input_shape=(args.channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    torch.cuda.set_device(gpu)
    generator.cuda(gpu)
    discriminator.cuda(gpu)
    feature_extractor.cuda(gpu)

    # define loss function (criterion) and optimizer
    criterion_GAN = nn.MSELoss().cuda(gpu)
    criterion_content = nn.L1Loss().cuda(gpu)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Save model attributes (erased after DDP)
    discriminator_output_shape = discriminator.output_shape

    # Wrap the model
    generator = nn.parallel.DistributedDataParallel(generator, device_ids=[gpu])
    discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[gpu], broadcast_buffers=False)
    feature_extractor = nn.parallel.DistributedDataParallel(feature_extractor, device_ids=[gpu])

    # Data loading code
    train_dataset = ImageDataset("../../data/%s" % args.dataset_name, hr_shape=hr_shape)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    torch.autograd.set_detect_anomaly(True)
    total_step = len(train_loader)

    if gpu == 0 :
        global_timer = Timer()
        epoch_timer = Timer()
        iter_timer = Timer()
        iter_time_meter = AverageMeter()

        global_timer.start()

    for epoch in range(args.n_epochs):
        if gpu == 0:
            epoch_timer.start()
        for i, imgs in enumerate(train_loader):
            if gpu == 0:
                iter_timer.start()
            imgs_lr = imgs["lr"].cuda(non_blocking=True)
            imgs_hr = imgs["hr"].cuda(non_blocking=True)

            valid = torch.ones((imgs_lr.size(0), *discriminator_output_shape), device=gpu)
            fake = torch.zeros((imgs_lr.size(0), *discriminator_output_shape), device=gpu)

            # ------------------
            #  Train Generators
            # ------------------
            gen_hr = generator(imgs_lr)

            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

            loss_D = (loss_real + loss_fake) / 2

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            if gpu == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] "
                    % (epoch, args.n_epochs, i, len(train_loader), loss_D.item(), loss_G.item()), end=''
                )
                iter_time_meter.update(iter_timer.stop())
                print('time for iteration: %.4f (%.4f)'%(iter_time_meter.val, iter_time_meter.avg))

                batches_done = epoch * len(train_loader) + i
                if batches_done % args.sample_interval == 0:
                    # Save image grid with upsampled inputs and SRGAN outputs
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                    imgs_hr_raw = imgs['hr_raw'].cuda(non_blocking=True)
                    with torch.no_grad():
                        print('[psnr] (imgs_lr):%.4f, (gen_hr):%.4f'%(psnr(minmaxscaler(imgs_lr), imgs_hr_raw, max_val=1).mean().item(), psnr(minmaxscaler(gen_hr), imgs_hr_raw, max_val=1).mean().item()))

                    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                    img_grid = torch.cat((imgs_hr_raw, imgs_lr, gen_hr), -1)
                    save_image(img_grid, "images/%d.png" % batches_done, normalize=False)
        if gpu==0:
            print('Elapsed_time for epoch(%s): %s'%(epoch, epoch_timer.stop()))
    if gpu == 0:
        print("Training complete in: %s "%global_timer.stop())
        print('Average time per iteration: %s'%str(iter_time_meter.avg))
        torch.save(generator.state_dict(), "saved_models/generator_%s.pth" % args.checkpoint_name)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%s.pth" % args.checkpoint_name)


if __name__ == '__main__':
    main()

'''
                 PSNR       SSIM
 generator    23.4061     0.7882
   low res    26.2965     0.7801

timer per iteration: 1.2152
'''
