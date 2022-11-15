#!/bin/sh
#SBATCH --job-name=ce_nres # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/slurm_%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
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

export MASTER_PORT=6143
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
# python compute_flops.py --dataset cifar10 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
# 	--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 \
#     --image_size 32 \
#     --batch_size 1 \
#     --current_resolution 32 \

python3 compute_flops.py --dataset celeba_512 --image_size 512 --exp ddgan_celebahq_512_exp1 --num_channels 3 --num_channels_dae 64 \
--ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 325 \
--current_resolution 512 \
--batch_size 1 \
# --measure_time \
# --two_gens \

 
# python3 compute_flops.py --dataset celeba_256 --image_size 256 --exp ddgan_p4_celebahq_exp2_glo --num_channels 3 --num_channels_dae 64 \
# --ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 800 \
# --current_resolution 256 \
# --patch_size 1 \
# --batch_size 1 \

# python3 test_ddgan.py --dataset lsun --image_size 256 --exp ddgan_lsun_exp1 --num_channels 3 --num_channels_dae 64 \
#     --ch_mult 1 1 2 2 4 4  --num_timesteps 4 --num_res_blocks 2  --epoch_id 500 \
#     --measure_time --batch_size 100 \

# python compute_flops.py --dataset stl10 --exp ddgan_stl10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
#     --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 900 \
#     --image_size 64 \
#     --current_resolution 64 \
#     --batch_size 1 \

# ----------------- Wavelet -----------
# python compute_flops.py --dataset cifar10 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
#  --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 \
#  --use_pytorch_wavelet \
#  --infer_mode both \
#  --image_size 32 \
#  --current_resolution 16 \
#  --attn_resolutions 32 \
#  --batch_size 1 \
#  # --net_type wavelet \
#  # --no_use_fbn \
#  # --magnify_data \
#  # --measure_time \


#1 1 2 2 4 4
# python3 compute_flops.py --dataset celeba_256 --image_size 256 --exp wddgan_celebahq_exp1_both128_atn16_recloss_wg12224_d5_nofreq_500ep --num_channels 3 --num_channels_dae 64 \
# --ch_mult 1 2 2 2 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 500 \
# --patch_size 1 --infer_mode both \
# --use_pytorch_wavelet \
# --current_resolution 128 \
# --attn_resolutions 16 \
# --batch_size 1 \
# --net_type wavelet \
# # --no_use_fbn \
# # --no_use_freq \
# # --no_use_residual \


# python3 compute_flops.py --dataset celeba_512 --image_size 512 --exp wddgan_celebahq_exp1_both256_atn16_recloss_new_wg112244_d5_400ep_skiphH --num_channels 3 --num_channels_dae 64 \
# --ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 200 \
# --patch_size 1 --infer_mode both \
# --use_pytorch_wavelet \
# --current_resolution 256 \
# --attn_resolutions 16 \
# --net_type wavelet \
# --batch_size 1 \
# # --two_gens \


# python3 test_wddgan.py --dataset lsun --image_size 256 --exp wddgan_lsun_exp1_wg12224_d5_400ep --num_channels 3 --num_channels_dae 64 \
# --ch_mult 1 2 2 2 4  --num_timesteps 4 --num_res_blocks 2  --epoch_id 400 \
# --infer_mode both \
# --use_pytorch_wavelet \
# --current_resolution 128 \
# --net_type wavelet \
# --measure_time \
# --batch_size 100 \
# # --compute_fid --compute_fid --real_img_dir pytorch_fid/lsun_church_stat.npy


# python3 test_wddgan.py --dataset ffhq_256 --image_size 256 --exp wddgan_ffhq_exp1_both128_400ep --num_channels 3 --num_channels_dae 64 \
# --ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 400 \
# --patch_size 1 --infer_mode both \
# --use_pytorch_wavelet \
# --current_resolution 128 \
# --attn_resolutions 16 \
# --compute_fid --real_img_dir pytorch_fid/ffhq_stat.npy \
# # --net_type wavelet \
# # --batch_size 100 \
# # --measure_time \
# # --two_gens \

# python compute_flops.py --dataset stl10 --exp wddgan_stl10_exp1_atn16_old_wg1222_d4_recloss_900ep --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
#     --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 600 \
#     --use_pytorch_wavelet \
#     --infer_mode both \
#     --image_size 64 \
#     --current_resolution 32 \
#     --attn_resolutions 16 \
#     --net_type wavelet \
#     --batch_size 1 \
#     # --compute_fid --real_img_dir pytorch_fid/stl10_stat.npy \
#     # --measure_time \




# --------------------- MULTISCALE WAVELET
# python train_multiscale_wddgan.py --dataset celeba_256 --image_size 256 --exp multiscale_wddgan_celebahq_exp5_hi64_recloss_maghloss_2step_g1222_d4_100ep --num_channels_dae 128 --num_timesteps 2 \
# --num_res_blocks 2 --batch_size 64 --num_epoch 100 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
# --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
# --current_resolution 64 \
# --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 1 \
# --train_mode only_hi \
# --use_pytorch_wavelet \
# --rec_loss \
# --num_disc_layers 4 \
# # --resume \
# # --net_type wavelet \


# python train_multiscale_wddgan_mag.py --dataset celeba_256 --image_size 256 --exp multiscale_wddgan_celebahq_exp6_hi64_recloss_maghloss_magh_2step_g1222_d3_200ep --num_channels_dae 64 --num_timesteps 2 \
# --num_res_blocks 2 --batch_size 32 --num_epoch 200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
# --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
# --current_resolution 64 \
# --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 1 \
# --train_mode only_hi \
# --use_pytorch_wavelet \
# --rec_loss \
# --num_disc_layers 3 \
# # --resume \
# # --net_type wavelet \

# multiscale_wddgan_celebahq_exp2_ll64_800ep/ multiscale_wddgan_celebahq_exp1_hi64/ multiscale_wddgan_celebahq_exp1_hi128
# multiscale_wddgan_celebahq_exp3_ll64_recloss_sg_sd4 multiscale_wddgan_celebahq_exp3_hi64_scaledgendisc_scalell_recloss_200ep multiscale_wddgan_celebahq_exp3_hi128_scalell_200ep
# multiscale_wddgan_celebahq_exp3_ll64_recloss_sg_sd4 multiscale_wddgan_celebahq_exp5_sg_hi64_recloss_1step_d4_200ep multiscale_wddgan_celebahq_exp5_g12224_hi128_recloss_1step_d4_200ep
# NEW
# multiscale_wddgan_celebahq_exp5_ll64_recloss_2step_g1222_d4_500ep multiscale_wddgan_celebahq_exp5_hi64_recloss_2step_g1222_d4_200ep multiscale_wddgan_celebahq_exp5_hi128_recloss_2step_g12224_d5_200ep

# multiscale_wddgan_celebahq_exp5_ll64_recloss_2step_g1222_d4_500ep multiscale_wddgan_celebahq_exp5_hi64_recloss_1step_g1222_d4_200ep multiscale_wddgan_celebahq_exp5_g12224_hi128_recloss_1step_d4_200ep

# multiscale_wddgan_celebahq_exp5_ll64_recloss_2step_g1222_d4_500ep multiscale_wddgan_celebahq_exp5_hi64_recloss_2step_g1222_d4_200ep multiscale_wddgan_celebahq_exp5_hi128_recloss_2step_g12224_d5_200ep

# python3 test_multiscale_wddgan.py --dataset celeba_256 --image_size 256  --num_res_blocks 2 \
# --exp multiscale_wddgan_celebahq_exp3_ll64_recloss_sg_sd4 multiscale_wddgan_celebahq_exp6_hi64_recloss_maghloss_magh_2step_g12224_d4_100ep multiscale_wddgan_celebahq_exp6_hi128_recloss_maghloss_magh_2step_g112244_d5_100ep \
# --epoch_id 500 100 100 \
# --num_timesteps 2 2 2 \
# --num_channels_dae 64 64 64 \
# --compute_fid --real_img_dir /lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/pytorch_fid/celebahq_stat.npy \
# # --batch_size 100 \
# # --measure_time \

# python3 test_multiscale_wddgan_mag.py --dataset celeba_256 --image_size 256  --num_res_blocks 2 \
# --exp multiscale_wddgan_celebahq_exp5_ll64_recloss_2step_g1222_d4_500ep multiscale_wddgan_celebahq_exp6_hi64_recloss_maghloss_magh_2step_g1222_d3_200ep multiscale_wddgan_celebahq_exp6_hi128_recloss_maghloss_magh_2step_g12224_d4_200ep \
# --epoch_id 500 200 200 \
# --num_timesteps 2 2 2 \
# --num_channels_dae 128 64 64 \
# --compute_fid --real_img_dir /lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/pytorch_fid/celebahq_stat.npy \
# # --batch_size 100 \
# # --measure_time \

  
