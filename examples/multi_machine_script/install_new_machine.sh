export http_proxy="http://192.168.0.140:8888"
export https_proxy="http://192.168.0.140:8888"

# 初始化
echo "TMOUT=86400" >> ~/.bashrc
source ~/.bashrc
echo "Host *\n    ServerAliveInterval 86400\n    ServerAliveCountMax 24" >> ~/.ssh/config

# 安装固件和驱动
#cp -r /home/save_dir/projects/install_for_new_machine/* /root
#cd /root
#./Ascend-hdk-910b-npu-driver_24.1.rc1_linux-aarch64.run  --full --quiet
#./Ascend-hdk-910b-npu-firmware_7.1.0.6.220.run --full

cp /YOUR_PATH/ubuntu_zsh.tar /YOUR_PATH/
ls /YOUR_PATH/ubuntu_zsh.tar
docker import /YOUR_PATH/ubuntu_zsh.tar ubuntu_zsh:1.0.0

# 安装docker工具
#cd /root
#tar zxf docker-19.03.10.tgz
#cp docker/* /usr/bin
#cat << EOF > /usr/lib/systemd/system/docker.service
#[Unit]
#Description=Docker Application Container Engine
#Documentation=https://docs.docker.com
#After=network-online.target firewalld.service
#Wants=network-online.target
#
#[Service]
#Type=notify
#ExecStart=/usr/bin/dockerd
#ExecReload=/bin/kill -s HUP \$MAINPID
#LimitNOFILE=infinity
#LimitNPROC=infinity
#TimeoutStartSec=0
#Delegate=yes
#KillMode=process
#Restart=on-failure
#StartLimitBurst=3
#StartLimitInterval=60s
#
#[Install]
#WantedBy=multi-user.target
#EOF
#
#sleep 1s
## 尝试start docker两次
#systemctl start docker
#sleep 5s
#systemctl start docker
#sleep 5s
#docker version
#
#ls /YOUR_PATH
#ls /YOUR_PATH
#ls /YOUR_PATH
#ls /YOUR_PATH
#ls /YOUR_PATH