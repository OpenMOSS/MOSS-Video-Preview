#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1
export WANDB_MODE=offline
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# export PATH=$PATH:"/usr/local/python3/bin:/usr/bin:/usr/bin/python3"
# export PATH="/root/anaconda3/bin:/root/anaconda3/condabin/conda:/root/anaconda3/bin/conda:$PATH"
# echo "$PATH"
# conda init
# conda activate py11
# source ~/.bashrc
echo "py11环境已激活"
#从云骁界面环境变量传入
NNODES=$(echo $NNODES)  


# k8s pytorchJob自动生成分布式训练相关环境变量
NODE_RANK=$(echo $RANK)
MASTER_ADDR=$(echo $MASTER_ADDR)
MASTER_PORT=$(echo $MASTER_PORT)
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


LOG_DIR="./logs/cwai-pretrain-llama2-7b-tp${TP}pp${PP}-${NNODES}nodes-log"

mkdir -p $LOG_DIR

FORCE_TORCHRUN=1 NNODES=${NNODES} RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
llamafactory-cli train /mnt/hpfs/streaming-video-llm-code/Streaming-Video-LLM/examples/train_full/llama3.2v_full_sft_hw_train_test.yaml \
2>&1 | tee ${LOG_DIR}/pretrain_llama2_7b_tp${TP}pp${PP}_${NNODES}_nodes_${NODE_RANK}_${TRAIN_STEPS}iters.log