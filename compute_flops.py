import argparse
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

from score_sde.models.ncsnpp_generator_adagn import NCSNpp, WaveletNCSNpp


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--measure_time', action='store_true', default=False,
                            help='whether or not measure time')
    parser.add_argument('--epoch_id', type=int, default=1000)
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
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)


    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')
        
    # wavelet GAN
    parser.add_argument("--use_pytorch_wavelet", action="store_true")
    parser.add_argument("--infer_mode", default="only_ll")
    parser.add_argument("--current_resolution", type=int, default=256)
    parser.add_argument("--net_type", default="normal")
    parser.add_argument("--magnify_data", action="store_true")
    parser.add_argument("--no_use_fbn", action="store_true")
    parser.add_argument("--no_use_freq", action="store_true")
    parser.add_argument("--no_use_residual", action="store_true")



    args = parser.parse_args()

    torch.manual_seed(42)
    device = 'cuda:0'


    if args.infer_mode == "only_ll":
        args.num_channels = 3 # low-res or wavelet-coefficients training
    elif args.infer_mode == "only_hi":
        args.num_channels = 9 # low-res or wavelet-coefficients training
    elif args.infer_mode == "both":
        args.num_channels = 12

    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    print(args.image_size, args.ch_mult, args.attn_resolutions)

    G_NET_ZOO = {"normal": NCSNpp, "wavelet": WaveletNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]
    print("GEN: {}".format(gen_net))


    netG = gen_net(args).to(device)
    netG.eval()

    mem = 0.
    for _ in range(300):

        x_t_1 = torch.randn(args.batch_size, args.num_channels, args.image_size, args.image_size).to(device)
        t = torch.randint(args.num_timesteps, (args.batch_size,)).to(device)

        latent_z = torch.randn(args.batch_size, args.nz, device=device)
        x_0 = netG(x_t_1, t, latent_z)
        mem += torch.cuda.max_memory_allocated(device) / 2**30
    print("Mem usage: {} (GB)".format(mem/300.))


    flops = FlopCountAnalysis(netG, (x_t_1, t, latent_z))
    print(flops.total())
    print(flop_count_table(flops))



