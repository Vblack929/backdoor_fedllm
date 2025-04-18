#!/bin/bash

python ../core/FL_text.py \
  --mode BD_baseline \
  --model bert \
  --epochs 3 \
  --local_ep 3 \
  --dataset sst2 \
  --tuning lora \
  --num_classes 2 \
  --num_users 20 \
  --frac 0.3 \
  --attackers 0.3 \
  --attack_type addSent \
  --lr 1e-4 \
  --optimizer adamw \
  --gpu \
  --defense krum