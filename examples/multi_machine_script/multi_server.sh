#!/bin/bash

rank=$1
n_rank=$2
main_process_ip=$3
FORCE_TORCHRUN=1 NNODES=${n_rank} RANK=${rank} MASTER_ADDR=${main_process_ip} MASTER_PORT=29500 \
llamafactory-cli train /home/save_dir/llama_test/LLaMA-Factory/examples/train_full/llama3.2v_full_sft_hw_train_test.yaml
