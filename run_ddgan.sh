#!/bin/sh
#SBATCH --job-name=gp_ce # create a short name for your job
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

export MASTER_PORT=6036
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


CURDIR=$(cd $(dirname $0); pwd)
echo 'The work dir is: ' $CURDIR

DATASET=$1
MODE=$2
GPUS=$3

if [ -z "$1" ]; then
   GPUS=1
fi

echo $DATASET $MODE

# ----------------- VANILLA -----------
if [[ $MODE == train ]]; then
	echo "==> Training DDGAN"

	if [[ $DATASET == cifar10 ]]; then
		python train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 2 --save_content --datadir ../data/cifar-10 --patch_size 1 \
			--master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node $GPUS \
			# --resume \
			# --use_local_loss \

	elif [[ $DATASET == celeba_256 ]]; then
		python train_ddgan.py --dataset celeba_256 --image_size 256 --exp ddgan_celebahq_256_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--patch_size 1 \
			--master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node $GPUS \
			# --resume
			# --use_local_loss \

	elif [[ $DATASET == celeba_512 ]]; then
		python train_ddgan.py --dataset celeba_512 --image_size 512 --exp ddgan_celebahq_512_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 2 --num_epoch 400 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir ../../celeba-lmdb-512/ \
			--patch_size 1 \
			--master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node $GPUS \
			--save_content_every 25 \
			# --resume
			# # --use_local_loss 

	elif [[ $DATASET == lsun ]]; then
		python3 train_ddgan.py --dataset lsun --image_size 256 --exp ddgan_lsun_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 8 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --save_content \
			--datadir data/lsun/ \
			--master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node $GPUS \
			# --resume 

	elif [[ $DATASET == ffhq_256 ]]; then
		python train_ddgan.py --dataset ffhq_256 --image_size 256 --exp ddgan_ffhq_256_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
			 --num_res_blocks 2 --batch_size 32 --num_epoch 600 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			 --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/ffhq/ffhq-lmdb/ \
			 --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node $GPUS \
	fi

else
	echo "==> Test DDGAN"

	if [[ $DATASET == cifar10 ]]; then
		python test_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 1200 \
			--patch_size 1 \
			--compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy 
			# --batch_size 100 \
			# --measure_time \

	elif [[ $DATASET == celeba_256 ]]; then
		python3 test_ddgan.py --dataset celeba_256 --image_size 256 --exp ddgan_celebahq_exp1 --num_channels 3 --num_channels_dae 64 \
			--ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 550 \
			--patch_size 1 \
			--compute_fid --real_img_dir /lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/pytorch_fid/celebahq_stat.npy
			# --batch_size 100 \
			# --measure_time \

	elif [[ $DATASET == celeba_512 ]]; then
		python3 test_ddgan.py --dataset celeba_512 --image_size 512 --exp ddgan_celebahq_512_exp1 --num_channels 3 --num_channels_dae 64 \
			--ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 325 \
			--batch_size 25 \
			--compute_fid --real_img_dir ./pytorch_fid/celebahq_512_stat.npy \
			# --measure_time \
			# --two_gens \

	elif [[ $DATASET == lsun ]]; then
		python3 test_ddgan.py --dataset lsun --image_size 256 --exp ddgan_lsun_exp1 --num_channels 3 --num_channels_dae 64 \
			--ch_mult 1 1 2 2 4 4  --num_timesteps 4 --num_res_blocks 2  --epoch_id 500 \
			--compute_fid --real_img_dir pytorch_fid/lsun_church_stat.npy \
			# --measure_time --batch_size 100 \

	elif [[ $DATASET == stl10 ]]; then
		python test_ddgan.py --dataset stl10 --exp ddgan_stl10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 900 \
			--image_size 64 \
			--compute_fid --real_img_dir pytorch_fid/stl10_stat.npy \
			# --batch_size 100 \
			# --measure_time \
	fi

fi # end mode
