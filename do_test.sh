#!/usr/bin/env bash

python run.py \
--model transformer \
--do_test \
--batch 64 \
--max_seq_length 256 \
--beam_search
