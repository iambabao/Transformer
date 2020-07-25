#!/usr/bin/env bash

python run.py \
--model transformer \
--do_train \
--do_eval \
--epoch 10 \
--batch 64 \
--max_seq_length 256 \
--optimizer custom \
--dropout 0.3 \
--pre_train_epochs 5 \
--early_stop 10
