#!/bin/bash
# Run multiple gaze prediction training scripts in parallel
# Usage:
#   python train.py --exp_name test --n_hidden 128 --lr 1e-3 --n_epochs 5 --device 'cuda:0'
# EPOCHS=30

# LR=1e-3
# GPU_ADDR='cuda:0'
# python train.py --exp_name nh8lr$LR --n_hidden 8 --lr $LR --n_epochs $EPOCHS --device $GPU_ADDR &
# python train.py --exp_name nh32lr$LR --n_hidden 32 --lr $LR --n_epochs $EPOCHS --device $GPU_ADDR &
# python train.py --exp_name nh128lr$LR --n_hidden 128 --lr $LR --n_epochs $EPOCHS --device $GPU_ADDR &

# LR=1e-2
# GPU_ADDR='cuda:1'
# python train.py --exp_name nh8lr$LR --n_hidden 8 --lr $LR --n_epochs $EPOCHS --device $GPU_ADDR &
# python train.py --exp_name nh32lr$LR --n_hidden 32 --lr $LR --n_epochs $EPOCHS --device $GPU_ADDR &
# python train.py --exp_name nh128lr$LR --n_hidden 128 --lr $LR --n_epochs $EPOCHS --device $GPU_ADDR &

# LR=1e-4
# GPU_ADDR='cuda:2'
# python train.py --exp_name nh8lr$LR --n_hidden 8 --lr $LR --n_epochs $EPOCHS --device $GPU_ADDR &
# python train.py --exp_name nh32lr$LR --n_hidden 32 --lr $LR --n_epochs $EPOCHS --device $GPU_ADDR &
# python train.py --exp_name nh128lr$LR --n_hidden 128 --lr $LR --n_epochs $EPOCHS --device $GPU_ADDR &

# search_deer task
EPOCHS=30
LR=1e-3
GPU_ADDR='cuda:0'
python train.py --exp_name nh64lr$LRgaze_search_deer --n_hidden 64 --lr $LR --n_epochs $EPOCHS --device $GPU_ADDR