#!/bin/bash

python FL_text.py \
  --mode BD_baseline \
  --model bert \
  --epochs 5 \
  --local_ep 5 \
  --dataset sst2 \
  --tuning lora \
  --num_classes 2 \
  --num_users 20 \
  --frac 0.1 \
  --lr 1e-5 \
  --optimizer adamw \
  --gpu \
  --save_model