# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import numpy as np
import copy

import os
import time

import torchvision
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from pytorch_fid.fid_score import calculate_fid_given_paths
from diffusion import *
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from train_multiscale_wddgan import cond_sample_from_model

def sample_from_cascaded_models(args, iwt, T, pos_coeff, generators, x_t_1_list, device):
    # generate ll first
    ll_sample = sample_from_model(pos_coeff[0], generators[0], args.num_timesteps_list[0], x_t_1_list[0], T[0], args) * (2*args.num_wavelet_levels)
    # print(ll_sample.min(), ll_sample.max())
    
    # then generate hi coeffs for IWT
    ll_list, hi_list = [], []
    for i, (netG, x_t_1) in enumerate(zip(generators[1:], x_t_1_list[1:])):
        scale = 2.*(args.num_wavelet_levels - i)
        ll_sample = ll_sample / scale # to [-1,1]
        ll_sample = torch.clamp(ll_sample, -1, 1)
        hi_sample = cond_sample_from_model(pos_coeff[i+1], netG, args.num_timesteps_list[i+1], x_t_1, T[i+1], args, cond=ll_sample)
    
        # scale to its original range
        ll_sample = ll_sample * scale 
        hi_sample = hi_sample * scale
        lh_sample = hi_sample[:, :3]
        hl_sample = hi_sample[:, 3:6]
        hh_sample = hi_sample[:, 6:9]

        ll_list.append(ll_sample)
        hi_list.append(hi_sample)

        # iwt
        ll_sample = iwt((ll_sample, [torch.stack((lh_sample, hl_sample, hh_sample), dim=2)]))
        # print(ll_sample.min(), ll_sample.max(), scale)
    
    # print(ll_sample.min(), ll_sample.max())
    fake_sample = torch.clamp(ll_sample, -1, 1)
    return fake_sample, ll_list, hi_list

#%%
def sample_and_test(args):
    torch.manual_seed(42)
    device = 'cuda:0'
    
    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celebahq_stat.npy'
    # elif args.dataset == 'lsun':
    #     real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    else:
        real_img_dir = args.real_img_dir


    assert args.num_wavelet_levels == (len(args.exp)-1)
    to_range_0_1 = lambda x: (x + 1.) / 2.

    CH_MULT = {
        32: [1, 2, 2, 2],
        64: [1, 2, 2, 2], # [1, 2, 2, 2, 4]
        128: [1, 2, 2, 2, 4], # [1, 1, 2, 2, 4, 4]
        256: [1, 1, 2, 2, 4, 4],
    }
    
    root_dir = './saved_info/multiscale_wdd_gan/{}'.format(args.dataset)
    generators = []
    resolutions = [args.image_size//2**i for i in range(args.num_wavelet_levels, 0, -1)]
    current_resolution = resolutions[0]

    for i in range(args.num_wavelet_levels+1):
        epoch_id = args.epoch_id[i]
        exp_path = args.exp[i]
        gen_args = copy.copy(args)
        print("Load pretrained generators at {} at {} epochs".format(exp_path, epoch_id))
        if "hi" not in exp_path:
            gen_args.num_channels = 3
            gen_args.num_out_channels = 3
            # gen_args.num_channels_dae = 128
        else: # hi
            gen_args.num_channels = 12
            gen_args.num_out_channels = 9 
            # gen_args.num_channels_dae = 64
        

        gen_args.image_size = current_resolution
        gen_args.ch_mult = CH_MULT[current_resolution]
        gen_args.num_channels_dae = args.num_channels_dae[i]
        # gen_args.ch_mult = CH_MULT[256]
        print(gen_args.ch_mult, gen_args.image_size)

        netG = NCSNpp(gen_args).to(device)

        # load weights
        ckpt_path = '{}/{}/netG_{}.pth'.format(root_dir, exp_path, epoch_id)
        ckpt = torch.load(ckpt_path, map_location=device)
        #loading weights from ddp in single gpu
        for key in list(ckpt.keys()):
            ckpt[key[7:]] = ckpt.pop(key)
        netG.load_state_dict(ckpt)
        netG.eval()

        generators.append(netG)
        if "hi" in exp_path:
            current_resolution *= 2
    
    iwt = DWTInverse(mode='zero', wave='haar').cuda()

    # T = get_time_schedule(args, device)
    T = []
    pos_coeff = []
    args.num_timesteps_list = args.num_timesteps
    for nT in args.num_timesteps_list:
        t_args = copy.deepcopy(args)
        t_args.num_timesteps = nT
        T.append(get_time_schedule(t_args, device))
        pos_coeff.append(Posterior_Coefficients(t_args, device))
    
        
    iters_needed = 50000 //args.batch_size
    
    save_dir = "./generated_multiscale_wavelets_samples/{}".format(args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.measure_time:
        x_t_1_list = [torch.randn(args.batch_size, 3, resolutions[0], resolutions[0]).to(device)] + [torch.randn(args.batch_size, 9, res, res).to(device) for res in resolutions]
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions,1))
        # GPU-WARM-UP
        for _ in range(10):
            _ = sample_from_cascaded_models(args, iwt, T, pos_coeff, generators, x_t_1_list, device)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = sample_from_cascaded_models(args, iwt, T, pos_coeff, generators, x_t_1_list, device)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print("Inference time: {:.2f}+/-{:.2f}ms".format(mean_syn, std_syn))
        exit(0)


    if args.compute_fid:
        for i in range(iters_needed):
            with torch.no_grad():
                x_t_1_list = [torch.randn(args.batch_size, 3, resolutions[0], resolutions[0]).to(device)] + [torch.randn(args.batch_size, 9, res, res).to(device) for res in resolutions]
                fake_sample, _, _ = sample_from_cascaded_models(args, iwt, T, pos_coeff, generators, x_t_1_list, device)
                fake_sample = to_range_0_1(fake_sample)
                for j, x in enumerate(fake_sample):
                    index = i * args.batch_size + j 
                    torchvision.utils.save_image(x, '{}/{}.jpg'.format(save_dir, index))
                print('generating batch ', i)
        
        paths = [save_dir, real_img_dir]
    
        kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))
    else:
        x_t_1_list = [torch.randn(args.batch_size, 3, resolutions[0], resolutions[0]).to(device)] + [torch.randn(args.batch_size, 9, res, res).to(device) for res in resolutions]
        fake_sample, ll_list, hi_list = sample_from_cascaded_models(args, iwt, T, pos_coeff, generators, x_t_1_list, device)
        fake_sample = to_range_0_1(fake_sample)
        
        scale = 2*args.num_wavelet_levels
        for i, (xll, xhi)  in enumerate(zip(ll_list, hi_list)):
            print("resolution {} with scale {}: ll range [{}, {}], hi range [{}, {}]".format(resolutions[i], scale, xll.min(), xll.max(), xhi.min(), xhi.max()))
            xll = xll / scale
            xll = torch.clamp(xll, -1, 1)
            xhi = xhi / scale
            
            torchvision.utils.save_image(to_range_0_1(xll), 'samples_ll{}_{}.jpg'.format(resolutions[i], args.dataset))
            torchvision.utils.save_image(to_range_0_1(xhi[:, :3]), 'samples_lh{}_{}.jpg'.format(resolutions[i], args.dataset))
            torchvision.utils.save_image(to_range_0_1(xhi[:, 3:6]), 'samples_hl{}_{}.jpg'.format(resolutions[i], args.dataset))
            torchvision.utils.save_image(to_range_0_1(xhi[:, 6:9]), 'samples_hh{}_{}.jpg'.format(resolutions[i], args.dataset))
            scale = scale / 2.

        torchvision.utils.save_image(fake_sample, 'samples_{}.jpg'.format(args.dataset))
        print("Results are saved at samples_{}.jpg".format(args.dataset))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--measure_time', action='store_true', default=False,
                            help='whether or not measure time')
    parser.add_argument('--epoch_id', type=int, nargs="+", default=1000)
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
    parser.add_argument('--num_channels_dae', type=int, nargs='+', default=(128,),
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
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
    parser.add_argument('--exp', default='experiment_cifar_default', nargs="+", help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, nargs='+', default=[4])

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')
        
    # wavelet GAN
    parser.add_argument("--num_wavelet_levels", default=2, type=int)
    parser.add_argument("--use_pytorch_wavelet", action="store_true")



    args = parser.parse_args()

    sample_and_test(args)
            
        
                        
