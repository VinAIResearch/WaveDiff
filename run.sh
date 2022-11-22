#!/bin/sh
export MASTER_PORT=6036
echo MASTER_PORT=${MASTER_PORT}

export PYTHONPATH=$(pwd):$PYTHONPATH

CURDIR=$(cd $(dirname $0); pwd)
echo 'The work dir is: ' $CURDIR

DATASET=$1
MODE=$2
GPUS=$3

if [ -z "$1" ]; then
   GPUS=1
fi

echo $DATASET $MODE $GPUS

# ----------------- Wavelet -----------
if [[ $MODE == train ]]; then
	echo "==> Training WaveDiff"

	if [[ $DATASET == cifar10 ]]; then
		python train_wddgan.py --dataset cifar10 --exp wddgan_cifar10_exp1_noatn_g122_d3_recloss_1800ep --num_channels 12 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3 --rec_loss \
			--use_pytorch_wavelet \

	elif [[ $DATASET == stl10 ]]; then
		python train_wddgan.py --dataset stl10 --image_size 64 --exp wddgan_stl10_exp1_atn16_wg1222_d4_recloss_900ep/ --num_channels 12 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 900 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 2 --save_content --datadir ./data/STL-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 32 --attn_resolutions 16 --num_disc_layers 4 --rec_loss \
			--net_type wavelet \
			--use_pytorch_wavelet \

	elif [[ $DATASET == celeba_256 ]]; then
		python train_wddgan.py --dataset celeba_256 --image_size 256 --exp wddgan_celebahq_exp1_atn16_wg12224_d5_recloss_500ep --num_channels 12 --num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 128 --attn_resolution 16 --num_disc_layers 5 --rec_loss \
			--save_content_every 10 \
			--net_type wavelet \
			# --use_pytorch_wavelet \

	elif [[ $DATASET == celeba_512 ]]; then
		python train_wddgan.py --dataset celeba_512 --image_size 512 --exp wddgan_celebahq512_exp1_atn16_wg112244_d6_recloss_400ep --num_channels 12 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 4 --num_epoch 400 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba_512/celeba-lmdb-512/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 256 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--net_type wavelet \
			--use_pytorch_wavelet \

	elif [[ $DATASET == lsun ]]; then
		python train_wddgan.py --dataset lsun --image_size 256 --exp wddgan_lsun_exp1_wg12224_d5_recloss_500ep --num_channels 12 --num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 16 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --save_content --datadir data/lsun/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 128 --attn_resolution 16 --num_disc_layers 5 --rec_loss \
			--save_content_every 10 \
			--net_type wavelet \
			--use_pytorch_wavelet \

	fi
else
	echo "==> Testing WaveDiff"
	if [[ $DATASET == cifar10 ]]; then \
		python test_wddgan.py --dataset cifar10 --exp wddgan_cifar10_exp1_noatn_g122_d3_recloss_1800ep --num_channels 12 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 --epoch_id 1300 \
			--image_size 32 --current_resolution 16 --attn_resolutions 32 \
			--use_pytorch_wavelet \
			# --compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy \
			# --batch_size 100 --measure_time \

	elif [[ $DATASET == stl10 ]]; then
		python test_wddgan.py --dataset stl10 --exp wddgan_stl10_exp1_atn16_wg1222_d4_recloss_900ep --num_channels 12 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 600 \
			--image_size 64 --current_resolution 32 --attn_resolutions 16 \
			--net_type wavelet \
			--use_pytorch_wavelet \
			# --compute_fid --real_img_dir pytorch_fid/stl10_stat.npy \
			# --batch_size 100 --measure_time \

	elif [[ $DATASET == celeba_256 ]]; then
		python3 test_wddgan.py --dataset celeba_256 --image_size 256 --exp wddgan_celebahq_exp1_atn16_wg12224_d5_recloss_500ep --num_channels 12 --num_channels_dae 64 \
			--ch_mult 1 2 2 2 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 475 \
			--current_resolution 128 --attn_resolutions 16 \
			--net_type wavelet \
			# --use_pytorch_wavelet \
			# --compute_fid --real_img_dir ./pytorch_fid/celebahq_stat.npy \
			# --batch_size 100 --measure_time \

	elif [[ $DATASET == celeba_512 ]]; then
		python3 test_wddgan.py --dataset celeba_512 --image_size 512 --exp wddgan_celebahq512_exp1_atn16_wg112244_d6_recloss_400ep --num_channels 12 --num_channels_dae 64 \
			--ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 350 \
			--current_resolution 256 --attn_resolutions 16 \
			--net_type wavelet \
			--use_pytorch_wavelet \
			# --compute_fid --real_img_dir pytorch_fid/celebahq_512_stat.npy \
			# --measure_time --batch_size 25 \

	elif [[ $DATASET == lsun ]]; then
		python3 test_wddgan.py --dataset lsun --image_size 256 --exp wddgan_lsun_exp1_wg12224_d5_recloss_500ep --num_channels 12 --num_channels_dae 64 \
			--ch_mult 1 2 2 2 4  --num_timesteps 4 --num_res_blocks 2  --epoch_id 400 \
			--current_resolution 128 --attn_resolutions 16 \
			--net_type wavelet \
			--use_pytorch_wavelet \
			# --compute_fid --compute_fid --real_img_dir pytorch_fid/lsun_church_stat.npy \
			# --batch_size 100 --measure_time \

	fi
fi
