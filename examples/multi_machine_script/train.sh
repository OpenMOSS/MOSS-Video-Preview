#!/bin/bash

rank=$1
n_rank=$2
main_process_ip=$3
local_process_ip=$4
project_name=$5
project_script=$6
proc_per_machine=8
num_processes=$((n_rank * proc_per_machine))
password=Fy!mATB@QE

# 初始检查
echo "rank=$1, n_rank=$2, main_process_ip=$3, local_process_ip=$4"
echo "Source env"
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/.bashrc
cd /home/save_dir/llama_test/LLaMA-Factory
# pip show torch_npu
# python --version

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

# 真实程序测试
#wandb login --relogin
#export WANDB_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
# export PROJECT=SoraClusterScripts${project_name}
#export ENTITY="opensora"
#export WANDB_MODE='offline'
# source /usr/local/Ascend/nnal/atb/set_env.sh
#python_path="/home/save_dir/projects/yancen/Open-Sora-Plan"
python_path="/usr/bin/python3"
# cd ${python_path}
export MAIN_PROCESS_IP_VALUE=${main_process_ip}
export NUM_MACHINE=${n_rank}
export MACHINE_RANK=${rank}
export NUM_PROCESSES=${num_processes}
export PROJECT_NAME=${project_name}
# export NUM_FRAME=${num_frame}
#export HCCL_EXEC_TIMEOUT=18000
export HCCL_CONNECT_TIMEOUT=3600
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=0
export PYTHONPATH=${python_path}:$PYTHONPATH
export PATH=$PATH:"/usr/local/python3/bin/"
echo "Killing process"
ps -ef | grep python | grep -v grep |awk '{print "kill -9 "$2}'|bash
sleep 10s    # 确保kill干净
# python3 --version

# Generate the new filename
echo "Start a new Running from here..."
new_filename="/home/save_dir/llama_test/LLaMA-Factory/examples/multi_machine_script/log_${project_name}.txt"
#new_filename=log_513_1.txt

echo "Start a new Running from here..." >> "$new_filename"
#rm -rf bind_core_*
#python3 bind_core.py &
sh ${project_script} ${rank} ${n_rank} ${main_process_ip} 2>&1 | tee -a $new_filename

# FORCE_TORCHRUN=1 NNODES=${n_rank} RANK=${rank} MASTER_ADDR=${main_process_ip} MASTER_PORT=29500 \
# llamafactory-cli train /home/save_dir/llama_test/LLaMA-Factory/examples/train_full/llama3.2v_full_sft_hw_train_test.yaml
#cat scripts/accelerate_configs/multi_node_example.yaml