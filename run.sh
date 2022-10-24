#!/bin/sh
#SBATCH --job-name=wgd # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/slurm_%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=80
#SBATCH --mem-per-gpu=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.haopt12@vinai.io

set -x
set -e

export MASTER_PORT=6081
export WORLD_SIZE=1
                                                                                              
export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
export SLURM_NODELIST=$SLURM_JOB_NODELIST
master_address=$(echo $SLURM_JOB_NODELIST | cut -d' ' -f1)
export MASTER_ADDRESS=$master_address

echo MASTER_ADDRESS=${MASTER_ADDRESS}                                                               
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${WORLD_SIZE}
echo "NODELIST="${SLURM_NODELIST}

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export PYTHONPATH=$(pwd):$PYTHONPATH


# ----------------- VANILLA -----------
# python -u -m torch.distributed.run --master_port=$MASTER_PORT --master_addr=$MASTER_ADDRESS --nproc_per_node=1 \
# python train_ddgan.py --dataset cifar10 --exp ddgan_p2_cifar10_glo_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
#     --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
#     --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
#     --ch_mult 1 2 2 2 --save_content --datadir ../data/cifar-10 --patch_size 2 \
#     --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 1 \
#     --resume \
#     # --use_local_loss \

# python train_ddgan.py --dataset celeba_256 --image_size 128 --exp ddgan_celebahq_128_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
# --num_res_blocks 2 --batch_size 32 --num_epoch 400 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
# --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
# --patch_size 1 \
# --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 1 \
# # --resume
# # --use_local_loss \

# python3 train_ddgan.py --dataset lsun --image_size 256 --exp ddgan_lsun_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 \
# --num_res_blocks 2 --batch_size 8 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
# --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 8 --save_content \
# --datadir data/lsun/ \
# --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 4 \
# --resume \

# python train_ddgan.py --dataset ffhq_256 --image_size 256 --exp ddgan_ffhq_256_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
#  --num_res_blocks 2 --batch_size 32 --num_epoch 600 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
#  --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/ffhq/ffhq-lmdb/ \
#  --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 4 \


# python test_ddgan.py --dataset cifar10 --exp ddgan_p2_cifar10_glo_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
# 	--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 1200 \
# 	--patch_size 1 \
#     --batch_size 100 \
#     --measure_time \
#     # --compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy 

# python3 test_ddgan.py --dataset celeba_256 --image_size 256 --exp ddgan_p4_celebahq_exp2_glo --num_channels 3 --num_channels_dae 64 \
# --ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 800 \
# --patch_size 1 \
# --batch_size 100 \
# --measure_time \
# # --compute_fid --real_img_dir /lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/pytorch_fid/celebahq_stat.npy



# ----------------- Wavelet -----------
# 1 2 2 2
# python train_wddgan.py --dataset cifar10 --exp wddgan_cifar10_exp1_atn8_g122_1000ep --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
#     --num_res_blocks 2 --batch_size 256 --num_epoch 1000 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
#     --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
#     --ch_mult 1 2 2 --save_content --datadir ../data/cifar-10 --patch_size 1 \
#     --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 1 \
#     --current_resolution 16 \
#     --attn_resolutions 8 \
#     --train_mode both \
#     --use_pytorch_wavelet \
#     --rec_loss \
#     --num_disc_layers 3 \
#     # --net_type wavelet
#     # --low_alpha 1. --high_alpha 2. \
#     # --two_disc \
#     # --resume

# wavelet: modify EMA swap...
# 1 1 2 2 4 4
python train_wddgan.py --dataset celeba_256 --image_size 256 --exp wddgan_celebahq_exp1_both128_atn8_recloss_wgd_400ep --num_channels 12 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
--num_res_blocks 2 --batch_size 32 --num_epoch 400 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
--current_resolution 128 \
--master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 1 \
--attn_resolution 8 \
--train_mode both \
--use_pytorch_wavelet \
--rec_loss \
--num_disc_layers 6 \
--net_type wavelet \
# --resume \
# --two_gens \
# --low_alpha 1. --high_alpha 2. \
# --two_disc \

# python train_wddgan.py --dataset lsun --image_size 256 --exp wddgan_lsun_exp2_300 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 \
# --num_res_blocks 2 --batch_size 32 --num_epoch 300 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
# --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 8 --save_content \
# --datadir data/lsun/ \
# --patch_size 1 --current_resolution 128 \
# --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 4 \
# --train_mode both \
# --use_pytorch_wavelet \
# --rec_loss \
# # --resume \
# # --low_alpha 1. --high_alpha 2. \
# # --two_disc \

# python train_wddgan.py --dataset ffhq_256 --image_size 256 --exp wddgan_ffhq_exp1_both128_400ep --num_channels 12 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
# --num_res_blocks 2 --batch_size 32 --num_epoch 400 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
# --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/ffhq/ffhq-lmdb/ \
# --current_resolution 128 \
# --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 4 \
# --train_mode both \
# --use_pytorch_wavelet \
# --attn_resolutions 16 \
# --rec_loss \
# # --resume \
# # --two_gens \
# # --low_alpha 1. --high_alpha 2. \
# # --two_disc \

 
# python test_wddgan.py --dataset cifar10 --exp wddgan_cifar10_exp1_atn8_1000ep --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
# 	--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 800 \
#     --use_pytorch_wavelet \
#     --infer_mode both \
#     --image_size 32 \
#     --current_resolution 16 \
#     --attn_resolutions 8 \
#     --compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy \
#     # --net_type wavelet \
#     # --batch_size 100 \
#     # --measure_time \

#wddgan_celebahq_both128_atn8_wgen_wdisc_800ep_exp1 
#1 1 2 2 4 4
# python3 test_wddgan.py --dataset celeba_256 --image_size 256 --exp  --num_channels 3 --num_channels_dae 64 \
# --ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 375 \
# --patch_size 1 --infer_mode both \
# --use_pytorch_wavelet \
# --current_resolution 128 \
# --attn_resolutions 8 \
# --compute_fid --real_img_dir /lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/pytorch_fid/celebahq_stat.npy \
# --net_type wavelet \
# # --batch_size 100 \
# # --measure_time \
# # --two_gens \


# python3 test_wddgan.py --dataset lsun --image_size 256 --exp wddgan_lsun_exp1 --num_channels 3 --num_channels_dae 64 \
# --ch_mult 1 1 2 2 4 4  --num_timesteps 4 --num_res_blocks 2  --epoch_id 400 \
# --infer_mode both \
# --use_pytorch_wavelet \
# --current_resolution 128 \
# --compute_fid --compute_fid --real_img_dir real_samples/lsun/ \


# --------------------- MULTISCALE WAVELET
# python train_multiscale_wddgan.py --dataset celeba_256 --image_size 256 --exp multiscale_wddgan_celebahq_exp4_wgd_ll64_recloss_800ep --num_channels_dae 64 --num_timesteps 2 \
# --num_res_blocks 2 --batch_size 128 --num_epoch 800 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
# --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
# --current_resolution 64 \
# --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 1 \
# --train_mode only_ll \
# --use_pytorch_wavelet \
# --rec_loss \
# --resume \
# --net_type wavelet \
# --num_disc_layers 4 \

# multiscale_wddgan_celebahq_exp2_ll64_800ep/ multiscale_wddgan_celebahq_exp1_hi64/ multiscale_wddgan_celebahq_exp1_hi128
# python3 test_multiscale_wddgan.py --dataset celeba_256 --image_size 256 --num_channels_dae 64 --num_timesteps 2 --num_res_blocks 2 \
# --exp multiscale_wddgan_celebahq_exp3_ll64_scaledgendisc_recloss_800ep multiscale_wddgan_celebahq_exp3_hi64_scaledgendisc_scalell_recloss_200ep multiscale_wddgan_celebahq_exp2_hi128_scalell_recloss_200ep  \
# --epoch_id 550 200 200 \
# --compute_fid --real_img_dir /lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/pytorch_fid/celebahq_stat.npy \
# # --batch_size 100 \
# # --measure_time \
  
