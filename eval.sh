#!/bin/sh
#SBATCH --job-name=gp_ls2 # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/quandm7/hao_workspace/DiffusionGAN/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/quandm7/hao_workspace/DiffusionGAN/slurm_%A.err # create a error file
#SBATCH --partition=applied # choose partition
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32 # 80
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

export MASTER_PORT=6034
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


# python compute_ddgan_fid.py --dataset stl10 --exp ddgan_stl10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
#     --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 900 \
#     --image_size 64 \
#     --compute_fid --real_img_dir pytorch_fid/stl10_stat.npy \
#     # --batch_size 100 \
#     # --measure_time \


# ----------------- Wavelet -----------
 
# python test_wddgan.py --dataset cifar10 --exp wddgan_cifar10_exp2_noatn_wg122_d3_recloss_bs64x4_1800ep --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
#     --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 --epoch_id 1400 \
#     --use_pytorch_wavelet \
#     --infer_mode both \
#     --image_size 32 \
#     --current_resolution 16 \
#     --attn_resolutions 32 \
#     --net_type wavelet \
#     --compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy \
#     # --batch_size 100 \
#     # --measure_time \


# python compute_fid.py --dataset stl10 --exp wddgan_stl10_exp1_atn16_old_wg1222_d4_recloss_900ep --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
#     --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 575 625 \
#     --use_pytorch_wavelet \
#     --infer_mode both \
#     --image_size 64 \
#     --current_resolution 32 \
#     --attn_resolutions 16 \
#     --compute_fid --real_img_dir pytorch_fid/stl10_stat.npy \
#     --net_type wavelet \
#     # --magnify_data \
#     # --batch_size 100 \
#     # --measure_time \

#1 1 2 2 4 4
# python3 compute_fid.py --dataset celeba_256 --image_size 256 --exp wddgan_celebahq_exp1_both128_atn16_recloss_wg12224_d5_500ep_skiphH --num_channels 3 --num_channels_dae 64 \
# --ch_mult 1 2 2 2 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 500 \
# --patch_size 1 --infer_mode both \
# --use_pytorch_wavelet \
# --current_resolution 128 \
# --attn_resolutions 16 \
# --compute_fid --real_img_dir /lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/pytorch_fid/celebahq_stat.npy \
# --net_type wavelet \
# # --no_use_fbn \
# # --batch_size 100 \
# # --measure_time \
# # --two_gens \


# python3 compute_fid.py --dataset celeba_512 --image_size 512 --exp wddgan_celebahq_exp1_both256_atn16_recloss_wg112244_d5_400ep_skiphH --num_channels 3 --num_channels_dae 64 \
# --ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 350 \
# --patch_size 1 --infer_mode both \
# --use_pytorch_wavelet \
# --current_resolution 256 \
# --attn_resolutions 16 \
# --net_type wavelet \
# --compute_fid --real_img_dir ./pytorch_fid/celebahq_512_stat.npy \
# --batch_size 100 \
# # --measure_time \
# # --two_gens \


python3 compute_fid.py --dataset lsun --image_size 256 --exp wddgan_lsun_exp1_wg12224_d5_500ep_bs128 --num_channels 3 --num_channels_dae 64 \
--ch_mult 1 2 2 2 4  --num_timesteps 4 --num_res_blocks 2  --epoch_id 25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500 \
--infer_mode both \
--use_pytorch_wavelet \
--current_resolution 128 \
--net_type wavelet \
--compute_fid --compute_fid --real_img_dir pytorch_fid/lsun_church_stat.npy


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
