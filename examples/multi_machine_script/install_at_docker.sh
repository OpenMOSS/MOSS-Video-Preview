#!/bin/zsh

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
#cd /YOUR_PATH/yancen

#EXPECTED_IP="192.168.0.140"
#CURRENT_IP=$(hostname -I | awk '{print $1}')
#
#if [[ "$CURRENT_IP" == "$EXPECTED_IP" ]]; then
#    unset http_proxy
#    unset https_proxy
#    tinyproxy
#else
#    export http_proxy="http://192.168.0.140:8888"
#    export https_proxy="http://192.168.0.140:8888"
#fi

#df -h

#cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
#cat /usr/local/Ascend/nnal/atb/latest/version.info

#cd /YOUR_PATH/projects/yancen/cann/910b_RC3
#./Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run --quiet --install
#./Ascend-cann-kernels-910b_8.0.RC3_linux-aarch64.run --quiet --install
#./Ascend-cann-nnal_8.0.RC3_linux-aarch64.run --quiet --install
#cd /YOUR_PATH/projects/yancen/MindSpeed
#pip install -e.

#pip install ftfy
#pip install beartype
#pip install timm==1.0.8
#pip install pandarallel
#pip install datasets
pip show datasets

#cd /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test/
#scp /YOUR_PATH/projects/yancen/hostfile ./
#make MPI_HOME=/usr/local/mpich-3.2.1 ASCEND_DIR=/usr/local/Ascend/ascend-toolkit/latest
#mpirun -f hostfile -n 384 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
#pip install pandarallel
#pip install datasets
#cd /home/save_dir/projects/yancen/MindSpeed
#pip install -e.
#pip install torch_npu==2.1.0.post3
#pip show torch_npu
#export HF_ENDPOINT=https://hf-mirror.com
#huggingface-cli download --repo-type dataset --resume-download BestWishYsh/ChronoMagic-ProH --local-dir-use-symlinks False --local-dir /YOUR_PATH/ChronoMagic-ProH

#ls /YOUR_PATH/
#cd /YOUR_PATH
#chmod +x *
#./Ascend-cann-toolkit_8.0.T16_linux-aarch64.run --full --quiet
#./Ascend-cann-kernels-910b_8.0.T16_linux.run --install --quiet
#cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

#pip3 install torch-npu==2.1.0.post3
#pip install /YOUR_PATH/lcalib-0.0.0-cp39-cp39-linux_aarch64.whl
#pip install "urllib3<1.26,>=1.20" botocore

#ps -ef | grep resize_vid | grep -v grep | awk '{print $2}' | xargs kill -9
#ps -ef | grep ffmpeg | grep -v grep | awk '{print $2}' | xargs kill -9
#sleep 5s
#
#ls /YOUR_PATH/resize_vid.py
#export PATH=/YOUR_PATH/bin/:$PATH
#python /YOUR_PATH/resize_vid.py --n_rank ${n_rank}


#docker commit opensora sora_plan:1.0.0
#
#pip3 show urllib3
#pip3 show torch_npu
#ls /YOUR_PATH
#ls /YOUR_PATH

#pip install torch_optimizer

#pip3 install deepspeed==0.12.6
