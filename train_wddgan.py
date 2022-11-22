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
from freq_utils import *

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
    # print(args.attn_resolutions)

    nz = args.nz #latent dimension
    if args.train_mode == "only_ll":
        args.num_channels = 3 # low-res or wavelet-coefficients training
    elif args.train_mode == "only_hi":
        args.num_channels = 9 # low-res or wavelet-coefficients training
    elif args.train_mode == "both":
        args.num_channels = 12

    dataset = create_dataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)
    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    G_NET_ZOO = {"normal": NCSNpp, "wavelet": WaveletNCSNpp}
    D_NET_ZOO = {"normal": [Discriminator_small, Discriminator_large], 
        "wavelet": [WaveletDiscriminator_small, WaveletDiscriminator_large]}
    gen_net = G_NET_ZOO[args.net_type]
    disc_net = D_NET_ZOO[args.disc_net_type]
    print("GEN: {}, DISC: {}".format(gen_net, disc_net))
    if args.two_gens:
        gen_args1 = copy.deepcopy(args)
        gen_args2 = copy.deepcopy(args)
        gen_args1.num_channels = 3
        gen_args2.num_channels = 12
        gen_args2.num_out_channels = 9
        netG = gen_net(gen_args1).to(device)
        netG_freq = gen_net(gen_args2).to(device)
    else:
        netG = gen_net(args).to(device)

    if args.dataset in ['cifar10', 'stackmnist', 'tiny_imagenet_200', 'slt10']:    
        if not args.two_disc:
           netD = disc_net[0](nc = 2*args.num_channels, ngf = args.ngf,
                                  t_emb_dim = args.t_emb_dim,
                                  act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)
        else:
           netD = disc_net[0](nc = 2*3, ngf = args.ngf,
                                   t_emb_dim = args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)
           netD_freq = disc_net[0](nc = 2*9, ngf = args.ngf,
                                  t_emb_dim = args.t_emb_dim,
                                  act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)
    else:
        if not args.two_disc:
            netD = disc_net[1](nc = 2*args.num_channels, ngf = args.ngf, 
                                      t_emb_dim = args.t_emb_dim,
                                      act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)
        else:
            netD = disc_net[1](nc = 2*3, ngf = args.ngf, 
                                       t_emb_dim = args.t_emb_dim,
                                       act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)
            netD_freq = disc_net[1](nc = 2*9, ngf = args.ngf, 
                                      t_emb_dim = args.t_emb_dim,
                                      act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)


    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=args.lr_g, betas = (args.beta1, args.beta2))


    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)

    if args.two_disc:
        broadcast_params(netD_freq.parameters())
        optimizerD_freq = optim.Adam(netD_freq.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
        schedulerD_freq = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD_freq, args.num_epoch, eta_min=1e-5)
    if args.two_gens:
        broadcast_params(netG_freq.parameters())
        optimizerG_freq = optim.Adam(netG_freq.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
        schedulerG_freq = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG_freq, args.num_epoch, eta_min=1e-5)
        if args.use_ema:
            optimizerG_freq = EMA(optimizerG_freq, ema_decay=args.ema_decay)


    #ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu], find_unused_parameters=False)
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])
    if args.two_disc:
        netD_freq = nn.parallel.DistributedDataParallel(netD_freq, device_ids=[gpu])
    if args.two_gens:
        netG_freq = nn.parallel.DistributedDataParallel(netG_freq, device_ids=[gpu])

    # Wavelet Pooling
    if not args.use_pytorch_wavelet:
        dwt = DWT_2D("haar")
        iwt = IDWT_2D("haar")
    else:
        dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
        iwt = DWTInverse(mode='zero', wave='haar').cuda()

    num_levels = int(np.log2(args.ori_image_size // args.current_resolution))


    exp = args.exp
    parent_dir = "./saved_info/wdd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir,exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))


    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume or os.path.exists(os.path.join(exp_path, 'content.pth')):
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

        # load D_freq
        if args.two_disc:
            netD_freq.load_state_dict(checkpoint['netD_freq_dict'])
            optimizerD_freq.load_state_dict(checkpoint['optimizerD_freq'])
            schedulerD_freq.load_state_dict(checkpoint['schedulerD_freq'])

        # load G_freq
        if args.two_gens:
            netG_freq.load_state_dict(checkpoint['netG_freq_dict'])
            optimizerG_freq.load_state_dict(checkpoint['optimizerG_freq'])
            schedulerG_freq.load_state_dict(checkpoint['schedulerG_freq'])

        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0


    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)
       
        for iteration, (x, y) in enumerate(data_loader):
            for p in netD.parameters():  
                p.requires_grad = True  
            netD.zero_grad()

            if args.two_disc:
                for p in netD_freq.parameters():  
                    p.requires_grad = True  
                netD_freq.zero_grad()

            for p in netG.parameters():  
                p.requires_grad = False  
            
            #sample from p(x_0)
            x0 = x.to(device, non_blocking=True)

            if not args.use_pytorch_wavelet:
                for i in range(num_levels):
                    xll, xlh, xhl, xhh = dwt(x0)
            else:
                xll, xh = dwt(x0) # [b, 3, h, w], [b, 3, 3, h, w]
                xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
            
            if args.train_mode == "only_ll":
                real_data = xll # [b, 3, h, w]
            elif args.train_mode == "only_hi":
                real_data = torch.cat([xlh, xhl, xhh], dim=1) # [b, 9, h, w]
            elif args.train_mode == "both":
                real_data = torch.cat([xll, xlh, xhl, xhh], dim=1) # [b, 12, h, w]
            
            
            # normalize real_data
            # real_data = (real_data - real_data.min()) / (real_data.max() - real_data.min()) # [0, 1]
            # real_data = real_data * 2 - 1 # [-1, 1]
            real_data = real_data / 2.0 # [-1, 1]
                
            assert -1 <= real_data.min() < 0
            assert 0 < real_data.max() <= 1

            #sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True
            

            # train with real
            if args.two_disc:
                xl_t, xh_t = x_t[:, :3], x_t[:, 3:]
                xl_tp1, xh_tp1 = x_tp1[:, :3], x_tp1[:, 3:]

                D_real = netD(xl_t, t, xl_tp1.detach()).view(-1)
                D_real_freq = netD_freq(xh_t, t, xh_tp1.detach()).view(-1)
                
                errD_real = F.softplus(-D_real)
                errD_real_freq = F.softplus(-D_real_freq)
                errD_real = args.low_alpha*errD_real.mean() + args.high_alpha*errD_real_freq.mean()
            else:
                D_real = netD(x_t, t, x_tp1.detach()).view(-1)
                errD_real = F.softplus(-D_real).mean()
            
            errD_real.backward(retain_graph=True)
            
            if args.lazy_reg is None:
                if args.two_disc:
                    grad_penalty_call(args, D_real, xl_t)
                    grad_penalty_call(args, D_real_freq, xh_t)
                else:
                    grad_penalty_call(args, D_real, x_t)
            else:
                if global_step % args.lazy_reg == 0:
                    if args.two_disc:
                        grad_penalty_call(args, D_real, xl_t)
                        grad_penalty_call(args, D_real_freq, xh_t)
                    else:
                        grad_penalty_call(args, D_real, x_t)

            # train with fake
            if args.two_gens:
                latent_z = torch.randn(batch_size, nz, device=device)
                latent_z_freq = torch.randn(batch_size, nz, device=device)
                xl_0_predict = netG(x_tp1[:, :3].detach(), t, latent_z)
                # xh_0_predict = netG_freq(x_tp1[:, 3:].detach(), t, latent_z_freq)
                xh_0_predict = netG_freq(torch.cat((xl_0_predict, x_tp1[:, 3:].detach()), dim=1), t, latent_z_freq)
                x_0_predict = torch.cat((xl_0_predict, xh_0_predict), dim=1)
            else:
                latent_z = torch.randn(batch_size, nz, device=device)
                x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            if args.two_disc:
                output = netD(x_pos_sample[:, :3], t, x_tp1[:, :3].detach()).view(-1)
                output_freq = netD_freq(x_pos_sample[:, 3:], t, x_tp1[:, 3:].detach()).view(-1)
                errD_fake = F.softplus(output)
                errD_fake_freq = F.softplus(output_freq) 
                errD_fake = errD_fake.mean() + errD_fake_freq.mean()
            else:
                output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
                errD_fake = F.softplus(output).mean()
            
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            if args.two_disc:
                optimizerD_freq.step()
            
        
            #update G
            for p in netD.parameters():
                p.requires_grad = False
            if args.two_disc:
                for p in netD_freq.parameters():
                    p.requires_grad = False

            for p in netG.parameters():  
                p.requires_grad = True  
            netG.zero_grad()
            if args.two_gens:
                netG_freq.zero_grad()
            
            
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
                
            
           
            if args.two_gens:
                latent_z = torch.randn(batch_size, nz,device=device)
                latent_z_freq = torch.randn(batch_size, nz,device=device)
                xl_0_predict = netG(x_tp1[:, :3].detach(), t, latent_z)
                # xh_0_predict = netG_freq(x_tp1[:, 3:].detach(), t, latent_z_freq)
                xh_0_predict = netG_freq(torch.cat((xl_0_predict, x_tp1[:, 3:].detach()), dim=1), t, latent_z_freq)
                x_0_predict = torch.cat((xl_0_predict, xh_0_predict), dim=1)
            else:
                latent_z = torch.randn(batch_size, nz,device=device)
                x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            if args.two_disc:
                output = netD(x_pos_sample[:, :3], t, x_tp1[:, :3].detach()).view(-1)
                output_freq = netD_freq(x_pos_sample[:, 3:], t, x_tp1[:, 3:].detach()).view(-1)
                errG = args.low_alpha*F.softplus(-output)
                errG_freq = args.high_alpha*F.softplus(-output_freq)
                errG = errG.mean() + errG_freq.mean()
            else:
                output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
                errG = F.softplus(-output).mean()
            
            # reconstructior loss
            if args.rec_loss:
                rec_loss = 0.
                rec_loss = rec_loss + F.l1_loss(x_0_predict, real_data)
                errG = errG + rec_loss
            
            errG.backward()
            optimizerG.step()
            if args.two_gens:
                optimizerG_freq.step()

            
            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(epoch,iteration, errG.item(), errD.item()))
        
        if not args.no_lr_decay:
            
            schedulerG.step()
            schedulerD.step()
            if args.two_disc:
                schedulerD_freq.step()
            if args.two_gens:
                schedulerG_freq.step()
        
        if rank == 0:
            if epoch % 10 == 0:
                x_pos_sample = x_pos_sample[:, :3]
                    
                torchvision.utils.save_image(x_pos_sample, os.path.join(exp_path, 'xpos_epoch_{}.png'.format(epoch)), normalize=True)
            
            x_t_1 = torch.randn_like(real_data)
            if args.two_gens:
                fake_sample = sample_from_dual_generators(pos_coeff, netG, netG_freq, args.num_timesteps, x_t_1, T, args)
            else:
                fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)

            if args.train_mode == "only_hi":
                fake_sample *= 2
                real_data *= 2
                if not args.use_pytorch_wavelet:
                    fake_sample = iwt(xll, fake_sample[:, :3], fake_sample[:, 3:6], fake_sample[:, 6:9])
                    real_data = iwt(xll, real_data[:, :3], real_data[:, 3:6], real_data[:, 6:9])
                else:
                    fake_sample = iwt((xll, [torch.stack((fake_sample[:, :3], fake_sample[:, 3:6], fake_sample[:, 6:9]), dim=2)]))
                    real_data = iwt((xll, [torch.stack((real_data[:, :3], real_data[:, 3:6], real_data[:, 6:9]), dim=2)]))
            
            elif args.train_mode == "both":
                fake_sample *= 2
                real_data *= 2
                if not args.use_pytorch_wavelet:
                     fake_sample = iwt(fake_sample[:, :3], fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12])
                     real_data = iwt(real_data[:, :3], real_data[:, 3:6], real_data[:, 6:9], real_data[:, 9:12])
                else:
                    fake_sample = iwt((fake_sample[:, :3], [torch.stack((fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))
                    real_data = iwt((real_data[:, :3], [torch.stack((real_data[:, 3:6], real_data[:, 6:9], real_data[:, 9:12]), dim=2)]))
            

            fake_sample = (torch.clamp(fake_sample, -1, 1) + 1)/2 # 0-1
            real_data = (torch.clamp(real_data, -1, 1) + 1)/2 # 0-1

            torchvision.utils.save_image(fake_sample, os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)))
            torchvision.utils.save_image(real_data, os.path.join(exp_path, 'real_data.png'))

            
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                               'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()} 
                    if args.two_disc:
                        content.update(
                                {'netD_freq_dict': netD_freq.state_dict(),
                               'optimizerD_freq': optimizerD_freq.state_dict(), 'schedulerD_freq': schedulerD_freq.state_dict()})
                    if args.two_gens:
                        content.update(
                                {'netG_freq_dict': netG_freq.state_dict(),
                               'optimizerG_freq': optimizerG_freq.state_dict(), 'schedulerG_freq': schedulerG_freq.state_dict()})
                    torch.save(content, os.path.join(exp_path, 'content.pth'))
                
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                    if args.two_gens:
                        optimizerG_freq.swap_parameters_with_ema(store_params_in_ema=True)
                    
                torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
                if args.two_gens:
                    torch.save(netG_freq.state_dict(), os.path.join(exp_path, 'netG_freq_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                    if args.two_gens:
                        optimizerG_freq.swap_parameters_with_ema(store_params_in_ema=True)

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
    parser.add_argument('--attn_resolutions', default=(16,), nargs='+', type=int,
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
    parser.add_argument("--two_disc", action="store_true")
    parser.add_argument("--low_alpha", type=float, default=1.)
    parser.add_argument("--high_alpha", type=float, default=1.)
    parser.add_argument("--use_pytorch_wavelet", action="store_true")
    parser.add_argument("--two_gens", action="store_true")
    parser.add_argument("--rec_loss", action="store_true")
    parser.add_argument("--net_type", default="normal")
    parser.add_argument("--disc_net_type", default="normal")
    parser.add_argument("--num_disc_layers", default=6, type=int)
    parser.add_argument("--no_use_fbn", action="store_true")
    parser.add_argument("--no_use_freq", action="store_true")
    parser.add_argument("--no_use_residual", action="store_true")

    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=50, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')

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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num_workers')


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

                
