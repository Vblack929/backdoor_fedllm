#!/bin/bash

python FL_text.py \
  --mode BD_baseline \
  --model bert \
  --epochs 1 \
  --local_ep 1 \
  --dataset sst2 \
  --tuning lora \
  --num_classes 2 \
  --num_users 20 \
  --frac 0.3 \
  --lr 1e-5 \
  --optimizer adamw