
CURRENT_IP=$(hostname -I | awk '{print $1}')
rank=$1
n_rank=$2
main_process_ip=$3
local_process_ip=$4

# 初始检查
#systemctl start docker
#docker start mindspeed_MM
#reboot
#echo "rank=$1, n_rank=$2, main_process_ip=$3, local_process_ip=$4"
# check
#npu-smi info
#df -h
#cd /YOUR_PATH/projects/yancen/cann
#./Ascend-cann-toolkit_8.0.T16_linux-aarch64.run --install --quiet
#./Ascend-cann-kernels-910b_8.0.T16_linux.run --install --quiet
#cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
#scp -r /YOUR_PATH/projects/yancen/mpich-3.2.1 /tmp
#cd /tmp
#wget https://ftp.gnu.org/gnu/automake/automake-1.15.tar.gz
#tar -xzvf automake-1.15.tar.gz
#cd automake-1.15
#./configure  --prefix=/opt/aclocal-1.15
#make
#sudo mkdir -p /opt
#sudo make install
#export PATH=/opt/aclocal-1.15/bin:$PATH
#aclocal --version
#cd ../mpich-3.2.1
#./configure --disable-fortran  --prefix=/usr/local/mpich-3.2.1
#make && make install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=$PATH:/usr/local/mpich-3.2.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/mpich-3.2.1/lib
which mpirun
cd /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test/
make MPI_HOME=/usr/local/mpich-3.2.1 ASCEND_DIR=/usr/local/Ascend/ascend-toolkit/latest
#mpirun -f hostfile -n 384 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
#docker stop mindspeed_MM
#docker rm mindspeed_MM
#docker rmi opensora:mindspeed_MM
#docker load -i /YOUR_PATH/projects/yancen/opensora_mindspeed_mm.tar
#docker run -itd --network host --privileged --ipc=host -u root  --device=/dev/davinci0  --device=/dev/davinci1  --device=/dev/davinci2  --device=/dev/davinci3  --device=/dev/davinci4  --device=/dev/davinci5  --device=/dev/davinci6  --device=/dev/davinci7  --device=/dev/davinci_manager  --device=/dev/devmm_svm  --device=/dev/hisi_hdc  -v /usr/slog:/usr/slog  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi  -v /etc/hccn.conf:/etc/hccn.conf  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver  -v /usr/slog:/usr/slog  -v /etc/ascend_install.info:/etc/ascend_install.info  --name=mindspeed_MM -v /YOUR_PATH/projects:/YOUR_PATH/projects -v /YOUR_PATH/obs_data:/YOUR_PATH/obs_data -v /YOUR_PATH:/YOUR_PATH opensora:mindspeed_MM /bin/bash

#du -sh /YOUR_PATH/video_data_obs/
#scp /YOUR_PATH/projects/yancen/panda70m_last_6268414_flowValue.json /YOUR_PATH/captions
# scp /YOUR_PATH/projects/yancen/Final_format_dataset_data_v2/step1.5_istock_final_612322_huawei.json /YOUR_PATH/captions/Final_format_dataset_data_v2

# bash /YOUR_PATH/projects/yancen/SoraClusterScripts/reboot.sh
#bash /YOUR_PATH/projects/yancen/SoraClusterScripts/mount_obs.sh

#ls /YOUR_PATH/captions
#du -sh /YOUR_PATH/projects_obs/images
#cd /YOUR_PATH/projects_obs/
#tar -zxvf images.tar.gz
#docker start opensora_cann_0627
#docker run -it --network host --privileged --name=opensora_cann_0627 --ipc=host -u root --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /YOUR_PATH:/YOUR_PATH -v /usr/slog:/usr/slog -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /etc/hccn.conf:/etc/hccn.conf -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/slog:/usr/slog -v /etc/ascend_install.info:/etc/ascend_install.info -v /YOUR_PATH:/YOUR_PATH -v /YOUR_PATH:/YOUR_PATH -v /YOUR_PATH:/YOUR_PATH -v /YOUR_PATH:/YOUR_PATH -v /YOUR_PATH/projects:/YOUR_PATH/projects -v /YOUR_PATH/video_data:/YOUR_PATH/video_data sora_plan:1.0.0 /bin/bash
#scp /YOUR_PATH/projects/yancen/civitai_v1_1940032.json  /YOUR_PATH/captions/
#cd /YOUR_PATH/captions/
#du -sh *
#rm -rf /YOUR_PATH/projects_obs/Images_ideogram_v1_n4
#du -sh /YOUR_PATH/projects_obs/civitai
#scp -r /YOUR_PATH/projects/yancen/Images_ideogram_v1 /YOUR_PATH/projects_obs/Images_ideogram_v1
#ps aux|grep download
#mkdir -p /YOUR_PATH/projects_obs/civitai
#/YOUR_PATH/obsutil_linux_arm64_5.5.12/obsutil cp obs://sora/20240426/20240425-storyblocks-5-9-ideogram-civitai-coverr/civitai/Images_civitai_v1 /YOUR_PATH/projects_obs/civitai -r -f
#/YOUR_PATH/obsutil_linux_arm64_5.5.12/obsutil cp obs://sora/20240426/MJ/images.tar.gz /YOUR_PATH/projects_obs/
#docker ps
#cp -r /YOUR_PATH/projects/mt5-xxl /YOUR_PATH/pre_weights/google
#resize2fs /dev/sdb
#df -h | grep /dev/sda2
#ls /YOUR_PATH/projects/checkpoints/image3d_240p_zp_umt5_from_initial_layer_28_head_24
#du -sh /YOUR_PATH/video_obs/panda70m/
#df -h
#docker ps
#npu-smi info
#ls /YOUR_PATH/video_obs/panda70m/
#ls /YOUR_PATH/video_obs/panda70m/
#ps aux|grep python
#reboot
#docker run -itd --network host --privileged --name=opensora_cann_5_25 --ipc=host -u root --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /YOUR_PATH:/YOUR_PATH -v /usr/slog:/usr/slog -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /etc/hccn.conf:/etc/hccn.conf -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/slog:/usr/slog -v /etc/ascend_install.info:/etc/ascend_install.info -v /YOUR_PATH:/YOUR_PATH -v /YOUR_PATH/projects:/YOUR_PATH/projects -v /YOUR_PATH/video_data:/YOUR_PATH/video_data sora_plan:1.0.0 /bin/bash
#docker commit opensora_cann_0627 sora_plan:3.0.0
#ps aux|grep python
#rm -rf /YOUR_PATH/yancen/Open-Sora-Plan-dev/pickles
#ls /YOUR_PATH/shebin/pre_weights/test140k
#rm -rf /YOUR_PATH/shebin/pre_weights/google/*
#scp /YOUR_PATH/projects/yancen/civitai_v1_1940032.json  /YOUR_PATH/captions/
#ps aux|grep python
#ls /YOUR_PATH/wwzhuo/Open-Sora-Plan-dev/taming
#docker ps
#npu-smi info
#ls /YOUR_PATH/wwzhuo/Open-Sora-Plan-dev/taming
#ls /YOUR_PATH/wwzhuo/Open-Sora-Plan-dev/scripts/text_condition/train_videoae_nx512x512.sh
#npu-smi info
#ls /YOUR_PATH/video_data_obs/pexel/5000-52/5881149.mp4
#ls /YOUR_PATH/yancen/Open-Sora-Plan-dev/pickles
#cat /YOUR_PATH/yancen/Open-Sora-Plan-dev/log_video3d_240p_zp_umt5_node8_layer_40_head_16.txt | grep "fault kernel_name=ReduceSum_c956995141babe6b46a0"
#df -h
#cd /YOUR_PATH/yancen/Open-Sora-Plan-dev
#npu-smi info
#du -sh /YOUR_PATH/projects_obs/images
#cd /YOUR_PATH
#find /YOUR_PATH/projects_obs/mj  -name "*.zip" | wc -l
#rm -rf /root/.obsutil*

#ps -ef | grep prepare_data.py | grep -v grep | awk '{print $2}' | xargs kill -9
#ps -ef | grep obsutil | grep -v grep | awk '{print $2}' | xargs kill -9

#cp /YOUR_PATH/projects/yancen/ubuntu_zsh.tar /YOUR_PATH/yancen/
#ls /YOUR_PATH/yancen/ubuntu_zsh.tar
#docker import /YOUR_PATH/yancen/ubuntu_zsh.tar ubuntu_zsh:1.0.0
#python /YOUR_PATH/projects/yancen/SoraClusterScripts/resize_vid.py

#sleep 5s
#cd /YOUR_PATH/yancen/Open-Sora-Plan-dev
#rm -rf opensora
#rm -rf scripts
#rm -rf vid_cap_list_64.pkl_local.pkl
#rm -rf vid_cap_list_513.pkl_local_f.pkl
#rm -rf npu_profiling_t2v
#rm -rf log*.txt
#rm -rf wandb

#echo $local_process_ip
#reboot
#docker commit opensora sora_plan:1.0.0


#grep -rn "high_performance" log.txt
#du -sh /YOUR_PATH/projects_obs/MJ_untar/images/
#mount /dev/nvme4n1 /YOUR_PATH
#ls pickles/vid_cap_list.pkl_local.pkl
#ls pickles/img_cap_list.pkl_local.pkl

#echo ${local_process_ip}
#reboot
#npu-smi info
#rm -rf wandb
#ls 512-0505/checkpoint-2500
#mkdir -p /YOUR_PATH/projects/checkpoints/512-pretrain-machine2/
#cp -r 512-0504/checkpoint-10500/ /YOUR_PATH/projects/checkpoints/512-pretrain-machine2/

#cp -r 512-0504/checkpoint-8500/ /YOUR_PATH/projects/checkpoints/512-pretrain-machine4/

#checkpoint_folder=/YOUR_PATH/projects/checkpoints/512-from-initial-machine4/
#mkdir -p $checkpoint_folder
#cp -r 512-0505/checkpoint-12000/ $checkpoint_folder

#scp -r root@192.168.0.118:/YOUR_PATH/yancen/Open-Sora-Plan-dev/wandb/latest-run/files /YOUR_PATH/projects/hzh/118/videos_1610
#!/bin/bash
#systemctl start docker
#docker commit opensora sora_plan:1.0.0
#scp -r root@192.168.0.78:/YOUR_PATH/yancen/Open-Sora-Plan-dev/wandb/latest-run/files /YOUR_PATH/projects/hzh/78/0506_0044
# 指定要遍历的根目录
#root_dir="/YOUR_PATH/yancen/Open-Sora-Plan-dev"
#
#
# 使用find命令查找所有名为checkpoint-xxx的文件夹
#find "$root_dir" -type d -name 'checkpoint-*' | while read -r dir; do
#    # 提取目录名中的数字
#    number=$(basename "$dir" | sed -e 's/checkpoint-//')
#
#    # 检查数字是否是整数
#    if ! [[ $number =~ ^[0-9]+$ ]]; then
#        echo "Skipping: $dir (not a valid number)"
#        continue
#    fi
#
#    # 计算模500的结果
#    mod=$(($number % 1000))
#
#    # 如果结果不为0，则删除该目录
#    if [ $mod -ne 0 ]; then
#        echo "Deleting: $dir"
#        rm -rf "$dir"
#    else
#        echo "Keeping: $dir"
#    fi
#done
#
#rm -rf /YOUR_PATH/yancen/5000-57

#python find_max.py
#echo "======================================================================" >> /YOUR_PATH/video_data/yancen/all_log.txt
#echo $local_process_ip >> /YOUR_PATH/video_data/yancen/all_log.txt
#grep -rn "step_loss" log_grad_max.txt | head -n 1 >> /YOUR_PATH/video_data/yancen/all_log.txt

#md5sum /YOUR_PATH/yancen/pre_weights/CausalVAEModel_4x8x8_0430/diffusion_pytorch_model.safetensors
#grep -rn step_loss log.txt
#grep -rn "Norm" log.txt
#rm -rf 512*
#ip_videos=/YOUR_PATH/video_data/yancen/$local_process_ip
#mkdir -p $ip_videos
#rm -rf $ip_videos
#cp -r /YOUR_PATH/yancen/Open-Sora-Plan-dev/wandb/latest-run/files/media/videos $ip_videos
#

#
## 进入指定目录
#directory=/YOUR_PATH/yancen/Open-Sora-Plan-dev/pickles
#cd "$directory"
#
## 遍历目录中所有符合模式的文件
#for file in vid_cap_list.pkl_*.pkl; do
#    if [[ $file =~ vid_cap_list\.pkl_([0-9]+)\.pkl ]]; then
#        # 提取数字部分
#        number="${BASH_REMATCH[1]}"
#        # 计算新的编号
#        new_number=$((number % 8))
#        # 构造新的文件名
#        new_file="vid_cap_list.pkl_${new_number}.pkl"
#        # 重命名文件
#        mv "$file" "$new_file"
#        echo "Renamed $file to $new_file"
#    fi
#done
#for file in img_cap_list.pkl_*.pkl; do
#    if [[ $file =~ img_cap_list\.pkl_([0-9]+)\.pkl ]]; then
#        # 提取数字部分
#        number="${BASH_REMATCH[1]}"
#        # 计算新的编号
#        new_number=$((number % 8))
#        # 构造新的文件名
#        new_file="img_cap_list.pkl_${new_number}.pkl"
#        # 重命名文件
#        mv "$file" "$new_file"
#        echo "Renamed $file to $new_file"
#    fi
#done
#mv /YOUR_PATH/video_data/mixkit /YOUR_PATH/video_data_obs

#yum install parallel

#cp /YOUR_PATH/AnyWord/ /YOUR_PATH/projects_obs
#mkdir -p /YOUR_PATH/projects_obs/MJ_untar
#cp /YOUR_PATH/MJ/images.tar.gz /YOUR_PATH/projects_obs/MJ
#cd /YOUR_PATH/projects_obs/
#mv MJ images.tar.gz
#mkdir -p /YOUR_PATH/projects_obs/MJ
#mv images.tar.gz /YOUR_PATH/projects_obs/MJ
#cd /YOUR_PATH/projects_obs/MJ
#target_path="/YOUR_PATH/projects_obs/MJ/images.tar.gz.file"
#find /YOUR_PATH/projects_obs/MJ -type f -name "images.tar.gz" -print0 |
#while IFS= read -r -d '' file; do
#    # 移动文件到目标路径
#    mv "$file" "$target_path"
#done
#du -h images.tar.gz
#file images.tar.gz
#rm -rf /YOUR_PATH/projects_obs/MJ/images.tar.gz
#tar --use-compress-program="pigz -p 20" -xpf $target_path -C /YOUR_PATH/projects_obs/MJ_untar

# 函数定义
#copy_zips() {
#    # 获取输入参数
#    local src_dir="$1"
#    local dest_dir="$2"
#
#    # 查找源目录下的所有 ZIP 文件
#    local zip_files
#    zip_files=$(find "$src_dir" -type f -name "*.zip" -maxdepth 2)
#
#    # 遍历每个 ZIP 文件
#    for zip_file in $zip_files; do
#        # 获取相对于源目录的路径
##        local relative_path="${zip_file#$src_dir/}"
#        relative_path=$(realpath --relative-to="$src_dir" "$zip_file")
#        echo $relative_path
#        # 构建目标文件的绝对路径
#        local dest_file="$dest_dir/${relative_path%/*}"
#        echo $dest_file
#        # 创建目标文件夹(如果不存在)
#        mkdir -p "$dest_file"
#
##        obs_source_file = ""
#        # 复制 ZIP 文件
#        cp "$zip_file" "$dest_file/"
##        /YOUR_PATH/obsutil_linux_arm64_5.5.12/obsutil cp "$zip_file" "$dest_file/"
#    done
#}
#
#rm -rf /YOUR_PATH/projects_obs/AnyWord-3M/
#copy_zips "/YOUR_PATH/AnyWord/AnyWord-3M/" "/YOUR_PATH/projects_obs/AnyWord-3M/"
#copy_zips "/YOUR_PATH/AnyWord/AnyWord-3M/ocr_data/" "/YOUR_PATH/projects_obs/AnyWord-3M/ocr_data/"
#copy_zips "/YOUR_PATH/AnyWord/AnyWord-3M/" "/YOUR_PATH/projects_obs/AnyWord-3M/"

#cp -r /YOUR_PATH/obsutil_linux_arm64_5.5.12 /YOUR_PATH
#umount /YOUR_PATH -l
#umount /YOUR_PATH -l
#systemctl stop nfs-server
#systemctl start nfs-server
#systemctl enable nfs-server
#
##
#mkdir /YOUR_PATH/projects/
#umount /YOUR_PATH/projects
#mount -t nfs 192.168.0.30:/YOUR_PATH/projects /YOUR_PATH/projects
###
#mkdir /YOUR_PATH/video_data
#umount /YOUR_PATH/video_data
#mount -t nfs 192.168.0.30:/YOUR_PATH/video_data /YOUR_PATH/video_data

#find /YOUR_PATH/yancen/Open-Sora-Plan-dev/ -name "*checkpoint*"
#systemctl start docker
#sleep 5s

#ls /YOUR_PATH
#ls /YOUR_PATH/video_data
#ls /YOUR_PATH/projects
#ls /YOUR_PATH/yancen/pre_weights
#ls /YOUR_PATH/sora_plan_v2.tar
#mkdir /YOUR_PATH/captions
#mv /YOUR_PATH/yancen/linbin_captions /YOUR_PATH/captions

#cat /usr/local/Ascend/driver/version.info
#npu-smi info
#curl www.baidu.com
#systemctl start docker
#reboot
#ls /YOUR_PATH/captions/linbin_captions
#ls /YOUR_PATH/captions/linbin_captions | wc -l

#ls /YOUR_PATH/captions/linbin_captions/video_pexel_65f_3832666.json
#ls /YOUR_PATH/captions/linbin_captions/video_pixabay_65f_601513.json
#ls /YOUR_PATH/captions/linbin_captions/video_mixkit_65f_54735.json


#mkfs.ext4 /dev/nvme4n1
#umount /YOUR_PATH -l
#mkdir /YOUR_PATH
##mount /dev/nvme4n1 /YOUR_PATH
#mkdir -p /YOUR_PATH/video_data/MJ/mixkit/mixkit
##echo "${local_process_ip}: copy /YOUR_PATH/video_pexel/5000-94 to /YOUR_PATH..."
#cp -r /YOUR_PATH/MJ/mixkit/mixkit/Woman /YOUR_PATH/video_data/MJ/mixkit/mixkit/
#cp -r /YOUR_PATH/MJ/mixkit/mixkit/Zoo /YOUR_PATH/video_data/MJ/mixkit/mixkit/
##echo "${local_process_ip}: copy file finished!"
#rm -rf /YOUR_PATH/video_pexel


