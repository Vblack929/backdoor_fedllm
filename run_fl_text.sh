#!/bin/bash

python FL_text.py \
  --mode BD_baseline \
  --model bert \
  --epochs 3 \
  --local_ep 5 \
  --dataset sst2 \
  --tuning lora \
  --num_classes 2 \
  --num_users 30 \
  --frac 0.3 \
  --attackers 0.3 \
  --attack_type addWord \
  --lr 1e-5 \
  --optimizer adamw \
  --gpu \
  --defense krum