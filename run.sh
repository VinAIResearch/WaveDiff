#!/bin/sh
#SBATCH --job-name=dg # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/slurm_%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=1
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

export MASTER_PORT=6111
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

# python -u -m torch.distributed.run --master_port=$MASTER_PORT --master_addr=$MASTER_ADDRESS --nproc_per_node=1 \
# python train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
#     --num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
#     --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
#     --ch_mult 1 2 2 2 --save_content --datadir ../data/cifar-10 --patch_size 1 \
#     --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 1


python test_ddgan.py --dataset cifar10 --exp ddgan_p4_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
	--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 1200 \
	--patch_size 2 --compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy 
