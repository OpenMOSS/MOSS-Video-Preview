#!/bin/bash

rank=$1
n_rank=$2
main_process_ip=$3
local_process_ip=$4
proc_per_machine=8
num_processes=$((n_rank * proc_per_machine))
password=Fy!mATB@QE

# 初始检查
echo "rank=$1, n_rank=$2, main_process_ip=$3, local_process_ip=$4"
source ~/.bashrc
cd /home/save_dir/projects/yancen

# EXPECTED_IP="192.168.0.140"
# CURRENT_IP=$(hostname -I | awk '{print $1}')

# if [[ "$CURRENT_IP" == "$EXPECTED_IP" ]]; then
#     unset http_proxy
#     unset https_proxy
#     tinyproxy
# else
#     export http_proxy="http://192.168.0.140:8888"
#     export https_proxy="http://192.168.0.140:8888"
# fi

export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=3

## 基础程序测试
cd /home/save_dir/llama_test/LLaMA-Factory/examples/multi_machine_script
pkill -9 python
sleep 4s    # 确保kill干净
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=${n_rank} --node_rank=${rank} --master_addr=${main_process_ip} --master_port=8822 test.py