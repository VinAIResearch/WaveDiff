# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import itertools
import argparse
import torch
import numpy as np
import copy

import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from datasets_prep.dataset import create_dataset

import shutil
import torch.distributed as dist
from torch.multiprocessing import Process

from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from diffusion import *
from utils import *


def cond_sample_from_model(coefficients, generator, n_time, x_init, T, opt, cond=None):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            if cond is None:
                x_0 = generator(x, t_time, latent_z)
            else:
                x_0 = generator(torch.cat([cond, x], dim=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
        
    return x

def grad_penalty_call(args, D_real, x_t):
    grad_real = torch.autograd.grad(
                outputs=D_real.sum(), inputs=x_t, create_graph=True
                )[0]
    grad_penalty = (
                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()
    
    
    grad_penalty = args.r1_gamma / 2 * grad_penalty
    grad_penalty.backward()

#%%
def train(rank, gpu, args):
    from score_sde.models.discriminator import Discriminator_small, Discriminator_large, WaveletDiscriminator_small, WaveletDiscriminator_large
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp, WaveletNCSNpp
    from EMA import EMA

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    nz = args.nz #latent dimension
    if args.train_mode == "only_ll":
        args.num_channels = 3 # low-res or wavelet-coefficients training
    elif args.train_mode == "only_hi":
        args.num_channels = 12 # 3 channels conditioning high-freq generation 
        args.num_out_channels = 9 
    elif args.train_mode == "both":
        args.num_channels = 12

    dataset = create_dataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)
    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    CH_MULT = {
        32: [1, 2, 2, 2],
        64: [1, 2, 2, 2, 4], # [1, 2, 2, 2], 
        128: [1, 1, 2, 2, 4, 4], #[1, 2, 2, 2, 4], 
        256: [1, 1, 2, 2, 4, 4],
    }
    # TODO: changing arch for each resolution
    # args.ch_mult = CH_MULT[args.ori_image_size] 
    args.ch_mult = CH_MULT[args.image_size] 

    G_NET_ZOO = {"normal": NCSNpp, "wavelet": WaveletNCSNpp}
    D_NET_ZOO = {"normal": [Discriminator_small, Discriminator_large], 
        "wavelet": [WaveletDiscriminator_small, WaveletDiscriminator_large]}
    gen_net = G_NET_ZOO[args.net_type]
    disc_net = D_NET_ZOO[args.net_type]
    print("GEN: {}, DISC: {}".format(gen_net, disc_net))

    netG = gen_net(args).to(device)

    if args.train_mode == "only_hi":
        num_in_channels = 9
    else:
        num_in_channels = 3

    if args.dataset == 'cifar10' or args.dataset == 'stackmnist':    
        netD = disc_net[0](nc = 2*num_in_channels, ngf = args.ngf,
                              t_emb_dim = args.t_emb_dim,
                              act=nn.LeakyReLU(0.2)).to(device)
    else:
        netD = disc_net[1](nc = 2*num_in_channels, ngf = args.ngf, 
                                  t_emb_dim = args.t_emb_dim,
                                  act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)


    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))


    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)


    #ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu], find_unused_parameters=True) # prevent error
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    # Wavelet Pooling
    num_levels = int(np.log2(args.ori_image_size // args.current_resolution))
    if not args.use_pytorch_wavelet:
        dwt = DWT_2D("haar")
        iwt = IDWT_2D("haar")
    else:
        dwt = DWTForward(J=num_levels, mode='zero', wave='haar').cuda()
        iwt = DWTInverse(mode='zero', wave='haar').cuda()

    exp = args.exp
    parent_dir = "./saved_info/multiscale_wdd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir,exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))


    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G
        
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])

        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    scale = 2.0**num_levels

    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)
       
        for iteration, (x, y) in enumerate(data_loader):
            for p in netD.parameters():  
                p.requires_grad = True  
            netD.zero_grad()

            #sample from p(x_0)
            x0 = x.to(device, non_blocking=True)

            if not args.use_pytorch_wavelet:
                for i in range(num_levels):
                    xll, xlh, xhl, xhh = dwt(x0)
            else:
                xll, xh = dwt(x0) # [b, 3, h, w], [b, 3, 3, h, w]
                xlh, xhl, xhh = torch.unbind(xh[-1], dim=2)
            
            if args.train_mode == "only_ll":
                real_data = xll # [b, 3, h, w]
            elif args.train_mode == "only_hi":
                real_data = torch.cat([xlh, xhl, xhh], dim=1) # [b, 9, h, w]
                xll = xll / scale # [-1, 1]
            elif args.train_mode == "both":
                real_data = torch.cat([xll, xlh, xhl, xhh], dim=1) # [b, 12, h, w]
            
            # normalize real_data
            # real_data = (real_data - real_data.min()) / (real_data.max() - real_data.min()) # [0, 1]
            # real_data = real_data * 2 - 1 # [-1, 1]
            real_data = real_data / scale # [-1, 1]
            # print(real_data.min(), real_data.max())
            assert -1 <= real_data.min() < 0
            assert 0 < real_data.max() <= 1
            # print("xll:", xll.min(), xll.max())
            # print("xlh:", xlh.min(), xlh.max())
            # print("xhl:", xhl.min(), xhl.max())
            # print("xhh:", xhh.min(), xhh.max())
            mag_real_data = real_data / torch.sqrt(torch.abs(real_data) + 0.000001) #torch.sign(real_data) * torch.pow(torch.abs(real_data)+0.0001, 0.5)
            
            #sample t
            t = torch.randint(0, args.num_timesteps, (mag_real_data.size(0),), device=device)
            
            x_t, x_tp1 = q_sample_pairs(coeff, mag_real_data, t)
            x_t.requires_grad = True
            

            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            errD_real = F.softplus(-D_real).mean()
            
            errD_real.backward(retain_graph=True)
            
            if args.lazy_reg is None:
                grad_penalty_call(args, D_real, x_t)
            else:
                if global_step % args.lazy_reg == 0:
                    grad_penalty_call(args, D_real, x_t)

            # train with fake
            
            # concatenate gt ll for condition high-freq generation
            latent_z = torch.randn(batch_size, nz, device=device)
            if args.train_mode == "only_hi":
                x_0_predict = netG(torch.cat((xll, x_tp1.detach()), dim=1), t, latent_z)
            else:
                x_0_predict = netG(x_tp1.detach(), t, latent_z)
            
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errD_fake = F.softplus(output).mean()
            
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            
        
            #update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            
            
            t = torch.randint(0, args.num_timesteps, (mag_real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, mag_real_data, t)

            latent_z = torch.randn(batch_size, nz,device=device)
            if args.train_mode == "only_hi":
                x_0_predict = netG(torch.cat([xll, x_tp1.detach()], dim=1), t, latent_z)
            else:
                x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errG = F.softplus(-output).mean()
            
            # reconstructior loss
            if args.rec_loss:
                ori_x_0_predict = x_0_predict * torch.abs(x_0_predict) # torch.sign(x_0_predict) * torch.pow(torch.abs(x_0_predict)+0.0001, 0.5)
                errG = errG + F.l1_loss(ori_x_0_predict, real_data) + F.l1_loss(x_0_predict, mag_real_data)

            
            errG.backward()
            optimizerG.step()
            
            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(epoch,iteration, errG.item(), errD.item()))
        
        if not args.no_lr_decay:
            
            schedulerG.step()
            schedulerD.step()
        
        if rank == 0:
            if epoch % 10 == 0:
                x_pos_sample = x_pos_sample[:, :3]
                ori_x_pos_sample = x_pos_sample * torch.abs(x_pos_sample)
                torchvision.utils.save_image(ori_x_pos_sample, os.path.join(exp_path, 'xpos_ori_epoch_{}.png'.format(epoch)), normalize=True)
                torchvision.utils.save_image(x_pos_sample, os.path.join(exp_path, 'xpos_epoch_{}.png'.format(epoch)), normalize=True)
            
            x_t_1 = torch.randn_like(real_data)
            if args.train_mode == "only_ll":
                fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)
            elif args.train_mode == "only_hi":
                lh, hl, hh = real_data[:, :3], real_data[:, 3:6], real_data[:, 6:9]
                mag_lh, mag_hl, mag_hh = mag_real_data[:, :3], mag_real_data[:, 3:6], mag_real_data[:, 6:9]
                fake_mag_sample = cond_sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args, cond=xll)
                fake_sample = fake_mag_sample * torch.abs(fake_mag_sample)
                flh, fhl, fhh = fake_sample[:, :3], fake_sample[:, 3:6], fake_sample[:, 6:9]
                fake_mag_lh, fake_mag_hl, fake_mag_hh = fake_mag_sample[:, :3], fake_mag_sample[:, 3:6], fake_mag_sample[:, 6:9]
                if not args.use_pytorch_wavelet:
                    fake_sample = iwt(scale*xll, scale*fake_sample[:, :3], scale*fake_sample[:, 3:6], scale*fake_sample[:, 6:9])
                    real_data = iwt(scale*xll, scale*real_data[:, :3], scale*real_data[:, 3:6], scale*real_data[:, 6:9])
                else:
                    fake_sample = iwt((scale*xll, [scale*torch.stack((fake_sample[:, :3], fake_sample[:, 3:6], fake_sample[:, 6:9]), dim=2)]))
                    real_data = iwt((scale*xll, [scale*torch.stack((real_data[:, :3], real_data[:, 3:6], real_data[:, 6:9]), dim=2)]))
                    # 64 -> 128: -2, 2, 128->256: -1, 1
            
            elif args.train_mode == "both":
                fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)
                fake_sample *= scale
                real_data *= scale
                if not args.use_pytorch_wavelet:
                     fake_sample = iwt(fake_sample[:, :3], fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12])
                     real_data = iwt(real_data[:, :3], real_data[:, 3:6], real_data[:, 6:9], real_data[:, 9:12])
                else:
                    fake_sample = iwt((fake_sample[:, :3], [torch.stack((fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))
                    real_data = iwt((real_data[:, :3], [torch.stack((real_data[:, 3:6], real_data[:, 6:9], real_data[:, 9:12]), dim=2)]))

            fake_sample = fake_sample * (2 / scale)
            real_data = real_data * (2 / scale)
            fake_sample = (torch.clamp(fake_sample, -1, 1) + 1)/2 # 0-1
            real_data = (torch.clamp(real_data, -1, 1) + 1)/2 # 0-1
            real_lh_data = (torch.clamp(lh, -1, 1) + 1)/2 # 0-1
            real_hl_data = (torch.clamp(hl, -1, 1) + 1)/2 # 0-1
            real_hh_data = (torch.clamp(hh, -1, 1) + 1)/2 # 0-1
            fake_lh_data = (torch.clamp(flh, -1, 1) + 1)/2 # 0-1
            fake_hl_data = (torch.clamp(fhl, -1, 1) + 1)/2 # 0-1
            fake_hh_data = (torch.clamp(fhh, -1, 1) + 1)/2 # 0-1
            real_mag_lh_data = (torch.clamp(mag_lh, -1, 1) + 1)/2 # 0-1
            real_mag_hl_data = (torch.clamp(mag_hl, -1, 1) + 1)/2 # 0-1
            real_mag_hh_data = (torch.clamp(mag_hh, -1, 1) + 1)/2 # 0-1
            fake_mag_lh_data = (torch.clamp(fake_mag_lh, -1, 1) + 1)/2 # 0-1
            fake_mag_hl_data = (torch.clamp(fake_mag_hl, -1, 1) + 1)/2 # 0-1
            fake_mag_hh_data = (torch.clamp(fake_mag_hh, -1, 1) + 1)/2 # 0-1

            torchvision.utils.save_image(fake_sample, os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)))
            torchvision.utils.save_image(fake_lh_data, os.path.join(exp_path, 'sample_discrete_lh_epoch_{}.png'.format(epoch)))
            torchvision.utils.save_image(fake_hl_data, os.path.join(exp_path, 'sample_discrete_hl_epoch_{}.png'.format(epoch)))
            torchvision.utils.save_image(fake_hh_data, os.path.join(exp_path, 'sample_discrete_hh_epoch_{}.png'.format(epoch)))
            torchvision.utils.save_image(real_data, os.path.join(exp_path, 'real_data.png'))
            torchvision.utils.save_image(real_lh_data, os.path.join(exp_path, 'real_lh.png'))
            torchvision.utils.save_image(real_hl_data, os.path.join(exp_path, 'real_hl.png'))
            torchvision.utils.save_image(real_hh_data, os.path.join(exp_path, 'real_hh.png'))
            torchvision.utils.save_image(real_mag_lh_data, os.path.join(exp_path, 'real_mag_lh.png'))
            torchvision.utils.save_image(real_mag_hl_data, os.path.join(exp_path, 'real_mag_hl.png'))
            torchvision.utils.save_image(real_mag_hh_data, os.path.join(exp_path, 'real_mag_hh.png'))
            torchvision.utils.save_image(fake_mag_lh_data, os.path.join(exp_path, 'fake_mag_lh.png'))
            torchvision.utils.save_image(fake_mag_hl_data, os.path.join(exp_path, 'fake_mag_hl.png'))
            torchvision.utils.save_image(fake_mag_hh_data, os.path.join(exp_path, 'fake_mag_hh.png'))
            
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                               'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()} 
                    torch.save(content, os.path.join(exp_path, 'content.pth'))
                
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                    
                torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true',default=False)

    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')


    parser.add_argument('--patch_size', type=int, default=1,
                            help='Patchify image into non-overlapped patches')
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), type=int, nargs='+',
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)

    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--datadir', default='./data')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    # wavelet GAN
    parser.add_argument("--current_resolution", type=int, default=256)
    parser.add_argument("--train_mode", default="only_ll")
    parser.add_argument("--use_pytorch_wavelet", action="store_true")
    parser.add_argument("--rec_loss", action="store_true")
    parser.add_argument("--net_type", default="normal")
    parser.add_argument("--num_disc_layers", default=6, type=int)


    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=50, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=2, help='save ckpt every x epochs')

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6002',
                        help='port for master')


    args = parser.parse_args()

    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        print('starting in debug mode')
        
        init_processes(0, size, train, args)

                
