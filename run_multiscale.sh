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

TYPE=$1
DATASET=$2
GPUS=$3
MODE=$4


# ----------------- VANILLA -----------
if [[ $TYPE =~ ddgan ]]; then
	if [[ $MODE =~ train ]]; then
		echo "==> Training DDGAN"
		if [[ $DATASET =~ cifar10 ]]; then
			python train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
			    --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			    --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			    --ch_mult 1 2 2 2 --save_content --datadir ../data/cifar-10 --patch_size 2 \
			    --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 1 \
			    # --resume \
			    # --use_local_loss \

		elif [[ $DATASET =~ celeba_256 ]]; then
			python train_ddgan.py --dataset celeba_256 --image_size 128 --exp ddgan_celebahq_128_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 400 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--patch_size 1 \
			--master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 1 \
			# --resume
			# --use_local_loss \

		elif [[ $DATASET =~ celeba_512 ]]; then
			python train_ddgan.py --dataset celeba_512 --image_size 512 --exp ddgan_celebahq_512_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 2 --num_epoch 400 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir ../../celeba-lmdb-512/ \
			--patch_size 1 \
			--master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 8 \
			--save_content_every 25 \
			# --resume
			# # --use_local_loss 

		elif [[ $DATASET =~ lsun ]]; then
			python3 train_ddgan.py --dataset lsun --image_size 256 --exp ddgan_lsun_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 8 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 8 --save_content \
			--datadir data/lsun/ \
			--master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 4 \
			# --resume 

		elif [[ $DATASET =~ ffhq_256 ]]; then
			python train_ddgan.py --dataset ffhq_256 --image_size 256 --exp ddgan_ffhq_256_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
			 --num_res_blocks 2 --batch_size 32 --num_epoch 600 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			 --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/ffhq/ffhq-lmdb/ \
			 --master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 4 \
		fi

	else
		echo "==> Test DDGAN"
		if [[ $DATASET =~ cifar10 ]]; then
			python test_ddgan.py --dataset cifar10 --exp ddgan_p2_cifar10_glo_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
				--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 1200 \
				--patch_size 1 \
			    --batch_size 100 \
			    --measure_time \
			    # --compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy 

		elif [[ $DATASET =~ celeba_256 ]]; then
			python3 test_ddgan.py --dataset celeba_256 --image_size 256 --exp ddgan_celebahq_exp1 --num_channels 3 --num_channels_dae 64 \
			--ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 550 \
			--patch_size 1 \
			--batch_size 100 \
			--compute_fid --real_img_dir /lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/pytorch_fid/celebahq_stat.npy
			# --measure_time \

		elif [[ $DATASET =~ celeba_512 ]]; then
			python3 test_ddgan.py --dataset celeba_512 --image_size 512 --exp ddgan_celebahq_512_exp1 --num_channels 3 --num_channels_dae 64 \
			--ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 325 \
			--batch_size 25 \
			--compute_fid --real_img_dir ./pytorch_fid/celebahq_512_stat.npy \
			# --measure_time \
			# --two_gens \

		elif [[ $DATASET =~ lsun ]]; then
			python3 test_ddgan.py --dataset lsun --image_size 256 --exp ddgan_lsun_exp1 --num_channels 3 --num_channels_dae 64 \
			    --ch_mult 1 1 2 2 4 4  --num_timesteps 4 --num_res_blocks 2  --epoch_id 500 \
			    --measure_time --batch_size 100 \

		elif [[ $DATASET =~ stl10 ]]; then
			python test_ddgan.py --dataset stl10 --exp ddgan_stl10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
			    --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 900 \
			    --image_size 64 \
			    --batch_size 100 \
			    --measure_time \
			    # --compute_fid --real_img_dir pytorch_fid/stl10_stat.npy \
		fi
	fi # end mode

else
	# ----------------- Wavelet -----------
	# 1 2 2 2
	if [[ $MODE =~ train ]]; then
		echo "==> Training WaveDiff"
		if [[ $DATASET =~ cifar10 ]]; then
			python train_wddgan.py --dataset cifar10 --exp wddgan_cifar10_exp2_noatn_wg122_d3_recloss_bs256_oldx2_1800ep --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
			    --num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			    --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			    --ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 --patch_size 1 \
			    --master_port $MASTER_PORT --num_process_per_node 1 \
			    --current_resolution 16 \
			    --attn_resolutions 32 \
			    --train_mode both \
			    --use_pytorch_wavelet \
			    --rec_loss \
			    --num_disc_layers 3 \
			    --net_type wavelet \
			    # --magnify_data \
			    # --disc_net_type wavelet \
			    # --resume
			    # --low_alpha 1. --high_alpha 2. \
			    # --two_disc \

		elif [[ $DATASET =~ stl10 ]]; then
			python train_wddgan.py --dataset stl10 --image_size 64 --exp wddgan_stl10_exp1_atn16_old_wg1222_d4_recloss_900ep --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
			    --num_res_blocks 2 --batch_size 256 --num_epoch 900 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			    --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			    --ch_mult 1 2 2 2 --save_content --datadir ./data/STL-10 --patch_size 1 \
			    --master_port $MASTER_PORT --num_process_per_node 1 \
			    --current_resolution 32 \
			    --attn_resolutions 16 \
			    --train_mode both \
			    --use_pytorch_wavelet \
			    --rec_loss \
			    --num_disc_layers 4 \
			    --net_type wavelet \
			    --resume
			    # --magnify_data \
			    # --disc_net_type wavelet \
			    # --low_alpha 1. --high_alpha 2. \
			    # --two_disc \


		elif [[ $DATASET =~ celeba_256 ]]; then
			# 1 1 2 2 4 4
			python train_wddgan.py --dataset celeba_256 --image_size 256 --exp wddgan_celebahq_exp1_both128_atn16_recloss_wg12224_oldx2_d5_500ep_skiphH --num_channels 12 --num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--current_resolution 128 \
			--master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 4 \
			--attn_resolution 16 \
			--train_mode both \
			--use_pytorch_wavelet \
			--num_disc_layers 5 \
			--net_type wavelet \
			--save_content_every 10 \
			--rec_loss \
			# --no_use_fbn \
			# --resume \
			# --magnify_data \
			# --disc_net_type wavelet \
			# --num_workers 4 \
			# --two_gens \
			# --low_alpha 1. --high_alpha 2. \
			# --two_disc \

		elif [[ $DATASET =~ celeba_512 ]]; then
			python train_wddgan.py --dataset celeba_512 --image_size 512 --exp wddgan_celebahq_exp1_both256_atn16_recloss_new_wg112244_d6_400ep_skiphH_bs16 --num_channels 12 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 4 --num_epoch 400 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir ../../celeba-lmdb-512/ \
			--current_resolution 256 \
			--master_port $MASTER_PORT --num_process_per_node 4 \
			--attn_resolution 16 \
			--train_mode both \
			--use_pytorch_wavelet \
			--rec_loss \
			--num_disc_layers 6 \
			--net_type wavelet \
			# --num_workers 4 \
			# --resume \
			# --two_gens \
			# --low_alpha 1. --high_alpha 2. \
			# --two_disc \
		elif [[ $DATASET =~ celeba_1024 ]]; then
			python train_wddgan.py --dataset celeba_1024 --image_size 1024 --exp wddgan_celebahq_exp1_both512_atn16_recloss_wg1248163232_d7_400ep_skiphH --num_channels 12 --num_channels_dae 16 --ch_mult 1 2 4 8 16 32 32 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 4 --num_epoch 400 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir ../../celeba-lmdb-1024/ \
			--current_resolution 512 \
			--master_port $MASTER_PORT --num_process_per_node 8 \
			--attn_resolution 16 \
			--train_mode both \
			--rec_loss \
			--num_disc_layers 7 \
			--net_type wavelet \
			--save_content_every 10 \
			# --num_workers 4 \
			# --resume \
			# --use_pytorch_wavelet \

		elif [[ $DATASET =~ lsun ]]; then
			python train_wddgan.py --dataset lsun --image_size 256 --exp wddgan_lsun_exp1_wg12224_d5_500ep_bs128 --num_channels 3 --num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 16 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --save_content --datadir data/lsun/ \
			--patch_size 1 --current_resolution 128 \
			--master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 8 --num_proc_node 1 \
			--train_mode both \
			--use_pytorch_wavelet \
			--num_disc_layers 5 \
			--rec_loss \
			--net_type wavelet \
			--save_content_every 10 \
			# --resume \
			# --magnify_data \
			# --disc_net_type wavelet \
			# --low_alpha 1. --high_alpha 2. \
			# --two_disc \

		elif [[ $DATASET =~ ffhq_256 ]]; then
			python train_wddgan.py --dataset ffhq_256 --image_size 256 --exp wddgan_ffhq_exp1_both128_atn16_recloss_wg12224_d5_500ep --num_channels 12 --num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 8 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/ffhq/ffhq-lmdb-256/ \
			--current_resolution 128 \
			--master_address $MASTER_ADDRESS --master_port $MASTER_PORT --num_process_per_node 8 \
			--train_mode both \
			--attn_resolutions 16 \
			--rec_loss \
			--num_disc_layers 5 \
			--net_type wavelet \
			--save_content_every 10 \
			# --resume \
			# --use_pytorch_wavelet \
		fi
	else
		echo "==> Testing WaveDiff"
		if [[ $DATASET =~ cifar10 ]]; then
			python test_wddgan.py --dataset cifar10 --exp wddgan_cifar10_exp2_noatn_wg122_d3_recloss_bs64x4_1800ep --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
			    --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 --epoch_id 1400 \
			    --use_pytorch_wavelet \
			    --infer_mode both \
			    --image_size 32 \
			    --current_resolution 16 \
			    --attn_resolutions 32 \
			    --net_type wavelet \
			    --compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy \
			    # --batch_size 100 \
			    # --measure_time \

		elif [[ $DATASET =~ celeba_256 ]]; then
			#1 1 2 2 4 4
			python3 test_wddgan.py --dataset celeba_256 --image_size 256 --exp wddgan_celebahq_exp1_both128_recloss_g12224_d5_500ep --num_channels 3 --num_channels_dae 64 \
			--ch_mult 1 2 2 2 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 500 \
			--patch_size 1 --infer_mode both \
			--use_pytorch_wavelet \
			--current_resolution 128 \
			--attn_resolutions 16 \
			--compute_fid --real_img_dir /lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/pytorch_fid/celebahq_stat.npy \
			# --net_type wavelet \
			# --no_use_fbn \
			# --batch_size 100 \
			# --measure_time \
			# --two_gens \

		elif [[ $DATASET =~ celeba_512 ]]; then
			python3 test_wddgan.py --dataset celeba_512 --image_size 512 --exp wddgan_celebahq_exp1_both256_atn16_recloss_wg112244_d5_400ep_skiphH --num_channels 3 --num_channels_dae 64 \
			--ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 350 \
			--patch_size 1 --infer_mode both \
			--use_pytorch_wavelet \
			--current_resolution 256 \
			--attn_resolutions 16 \
			--net_type wavelet \
			--compute_fid --real_img_dir ./pytorch_fid/celebahq_512_stat.npy \
			--batch_size 100 \
			# --measure_time \
			# --two_gens \

		elif [[ $DATASET =~ lsun ]]; then
			python3 test_wddgan.py --dataset lsun --image_size 256 --exp wddgan_lsun_exp1_wg12224_d5_500ep --num_channels 3 --num_channels_dae 64 \
			--ch_mult 1 2 2 2 4  --num_timesteps 4 --num_res_blocks 2  --epoch_id 500 \
			--infer_mode both \
			--use_pytorch_wavelet \
			--current_resolution 128 \
			--net_type wavelet \
			--compute_fid --compute_fid --real_img_dir pytorch_fid/lsun_church_stat.npy
			# --measure_time \
			# --batch_size 100 \

		elif [[ $DATASET =~ ffhq_256 ]]; then
			python3 test_wddgan.py --dataset ffhq_256 --image_size 256 --exp wddgan_ffhq_exp1_both128_400ep --num_channels 3 --num_channels_dae 64 \
			--ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 400 \
			--patch_size 1 --infer_mode both \
			--use_pytorch_wavelet \
			--current_resolution 128 \
			--attn_resolutions 16 \
			--compute_fid --real_img_dir pytorch_fid/ffhq_stat.npy \
			# --net_type wavelet \
			# --batch_size 100 \
			# --measure_time \
			# --two_gens \

		elif [[ $DATASET =~ stl10 ]]; then
			python test_wddgan.py --dataset stl10 --exp wddgan_stl10_exp1_atn16_old_wg1222_d4_recloss_900ep --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
			    --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 600 \
			    --use_pytorch_wavelet \
			    --infer_mode both \
			    --image_size 64 \
			    --current_resolution 32 \
			    --attn_resolutions 16 \
			    --net_type wavelet \
			    --compute_fid --real_img_dir pytorch_fid/stl10_stat.npy \
			    # --measure_time \
			    # --batch_size 100 \
		fi

	fi # end mode

fi # end type

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
# --exp multiscale_wddgan_celebahq_exp3_ll64_recloss_sg_sd4 multiscale_wddgan_celebahq_exp6_hi64_recloss_maghloss_magh_2step_g12224_d4_100ep multiscale_wddgan_celebahq_exp6_hi128_recloss_maghloss_magh_2step_g112244_d5_100ep \
# --epoch_id 500 100 100 \
# --num_timesteps 2 2 2 \
# --num_channels_dae 64 64 64 \
# # --compute_fid --real_img_dir /lustre/scratch/client/vinai/users/haopt12/DiffusionGAN/pytorch_fid/celebahq_stat.npy \
# # --batch_size 100 \
# # --measure_time \
