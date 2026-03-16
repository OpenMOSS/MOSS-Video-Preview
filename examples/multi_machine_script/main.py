import os
import random
import tarfile
import threading
import multiprocessing
import argparse
from pathlib import Path
from fabric import Connection, task
from itertools import islice
import json



def scp_files(conn, host, files):
    conn.run(f"mkdir -p {script_base_path}", warn=True)
    for file in files:
        script_path = f"{script_base_path}/{file}"
        os.system(f"scp -r {script_path} root@{host}:{os.path.dirname(script_path)}")

def get_host_file_lists():
    tar_files = []

    def find_tar_files(root_dir):
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.tar'):
                    tar_files.append(os.path.join(root, file))

    find_tar_files("/YOUR_PATH/download_sam_600_999")
    find_tar_files("/YOUR_PATH/sa")
    tar_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0][3:]))

    def distribute_files(tar_files, n):
        # 计算每个主机最少分配到的文件数
        min_files_per_host = len(tar_files) // n
        # 计算不能均匀分配的剩余文件数
        remainder = len(tar_files) % n

        # 初始化结果列表，每个元素代表一个主机分配到的文件列表
        result = []
        start_index = 0

        for i in range(n):
            # 如果剩余文件数大于0，这个主机分配一个额外的文件
            files_count = min_files_per_host + (1 if i < remainder else 0)
            # 从tar_files中取出相应数量的文件·
            end_index = start_index + files_count
            # 将这部分文件加入到结果列表
            result.append(tar_files[start_index:end_index])
            # 更新下一个分配的起始索引
            start_index = end_index

        return result

    host_file_lists = distribute_files(tar_files, len(hosts))
    sum_files = 0
    for item in host_file_lists:
        sum_files += len(item)
    assert sum_files == len(tar_files)
    return host_file_lists

# 其他机器免密登录本机器函数
def setup_ssh_key_for_others(conn):
    if conn.host == f'root@{local_ip}':
        return

    result = conn.run("ls ~/.ssh/id_rsa", warn=True).stdout
    conn.run('echo N | ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa', warn=True)

    # 将公钥添加到远程主机的 authorized_keys 文件中
    public_key = conn.run('cat ~/.ssh/id_rsa.pub', hide=True).stdout

    os.system(f'echo "{public_key}" >> ~/.ssh/authorized_keys')

    print(f'已在主机 {conn.host} 上设置免密登录，其key为{public_key}')


# 本机器免密登录其他机器函数
def setup_ssh_key_for_local():
    # 生成 SSH 密钥对
    if not os.path.exists('/root/.ssh/id_rsa.pub'):
        print(f"No ~/.ssh/id_rsa.pub of current host {local_ip}")
        os.system('ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa')

    # 获取本地公钥内容
    with open(os.path.expanduser('~/.ssh/id_rsa.pub'), 'r') as f:
        public_key = f.read().strip()

    # 将公钥添加到远程主机的 authorized_keys 文件中
    for host in hosts:
        if host == local_ip:
            continue
        print(host)
        print(password)
        conn = Connection(f"root@{host}", connect_kwargs={'password': password})
        conn.run(f'echo "{public_key}" >> ~/.ssh/authorized_keys')
        print(f'已在主机 {conn.host} 上设置免密登录')


def mount(conn, host):
    commands = ["yum install nfs-utils",
                "systemctl start nfs-server",
                "systemctl enable nfs-server",
                "mkdir -p /home/docker",
                "mount /dev/nvme2n1p1 /home/docker",
                # "mkdir -p /home/save_dir/projects/",
                # "mount -t nfs 192.168.0.30:/home/save_dir/projects /home/save_dir/projects",
                # "mkdir -p /home/video_data",
                # "mount -t nfs 192.168.0.30:/home/video_data /home/video_data",
                "mkdir -p /home/save_dir",
                "mount -t nfs -o vers=3,nolock 192.168.0.248:/  /home/save_dir"
                ]

    for command in commands:
        conn.run(command, pty=True, warn=True)


def run_docker_base(conn, host, rank, n_rank, script_path, restart_docker=False):
    # 1. 判断容器是否已经启动,如果没有启动则执行 docker start opensora
    result = conn.run(f'docker inspect --format="{{.State.Running}}" {docker_name}', warn=True)
    if 'false' in result.stdout:
        print(f'启动容器 {docker_name} 在主机 {conn.host}')
        conn.run(f'docker start {docker_name}')

    # 2. 基于zsh 进入 opensora容器
    # if host != local_ip:
    #     conn.run(f"rm {script_path}", warn=True)
    # os.system(f"scp -r /home/opensora/sora_plan_v2.tar root@{host}:/home/opensora/")
    if restart_docker:
        print(f"{host}: stop docker")
        conn.run(f"docker stop {docker_name} -t 10", warn=True)
        conn.run("systemctl start docker", warn=True)
        print(f"{host}: start docker")
        conn.run(f"docker start {docker_name}", warn=True)

    print(f"{'-' * 60} Enter docker of {host} {'-' * 60}")
    # conn.run(f"mkdir -p {base_path}", warn=True)
    # conn.run(f"mkdir -p {os.path.dirname(script_path)}", warn=True)

# 任务函数
@task
def test_connection(conn, host, rank, n_rank):
    script_path = "/YOUR_PATH/test_connection.sh"
    run_docker_base(conn, host, rank, n_rank, script_path)
    conn.run(f'docker exec {docker_name} bash {script_path} {rank} {n_rank} {main_process_ip} {host}')


@task
def train(conn, host, rank, n_rank):
    project_name = args.project_name
    project_script = args.project_script
    num_frame = args.num_frame

    # assert project_script is not None
    # assert project_name is not None
    # assert num_frame is not None

    script_path = "/YOUR_PATH/train.sh"
    # for folder in ["opensora", "scripts", "taming"]:
    #     os.system(f"scp -r {base_path}/{folder} root@{host}:{base_path}/")

    run_docker_base(conn, host, rank, n_rank, script_path, restart_docker=True)
    conn.run(f'docker exec {docker_name} bash {script_path} {rank} {n_rank} {main_process_ip} {host} {project_name} {project_script}', pty=True)

@task
def test(conn, host, rank, n_rank):
    project_name = args.project_name
    project_script = args.project_script
    num_frame = args.num_frame

    # assert project_script is not None
    # assert project_name is not None
    # assert num_frame is not None
    script_path = "/YOUR_PATH/test.sh"
    # for folder in ["opensora", "scripts", "taming", "examples"]:
    #     os.system(f"scp -r {base_path}/{folder} root@{host}:{base_path}/")

    run_docker_base(conn, host, rank, n_rank, script_path)
    conn.run(f'docker exec {docker_name} bash {script_path} {rank} {n_rank} {main_process_ip} {host} {project_name} {project_script} {num_frame} {args.height} {args.width}', pty=True)


@task
def print_latest_log(conn, host, rank, n_rank):
    project_name = args.project_name
    assert project_name is not None
    script_path = "/YOUR_PATH/tail_log.sh"
    conn.run(f'docker exec {docker_name} bash {script_path} {rank} {n_rank} {main_process_ip} {host} {project_name}', pty=True, warn=True)


@task
def kill(conn, host, rank, n_rank):
    script_path = "/YOUR_PATH/kill.sh"
    # run_docker_base(conn, host, rank, n_rank, script_path, restart_docker=False)
    # conn.run(f'docker exec opensora bash {script_path} {rank} {n_rank} {main_process_ip} {host}', pty=True, warn=True)
    # conn.run(f'docker exec ubuntu zsh {script_path} {rank} {n_rank} {main_process_ip} {host}', pty=True, warn=True)
    # run_docker_base(conn, host, rank, n_rank, script_path, restart_docker=True)
    conn.run(f"bash {script_path} {rank} {n_rank} {main_process_ip} {host}", pty=True, warn=True)


@task
def reboot_npu(conn, host, rank, n_rank):
    if host == local_ip:
        return
    # script_path = "/home/save_dir/projects/yancen/SoraClusterScripts/reboot_npu.sh"
    mount(conn, host)
    # conn.run(f"bash {script_path} {rank} {n_rank} {main_process_ip} {host}", pty=True, warn=True)
    # install_docker(conn, host, rank, n_rank)


@task
def install_at_local(conn, host, rank, n_rank):
    if host == local_ip:
        return
    project_name = args.project_name
    # assert project_name is not None
    script_path = "/YOUR_PATH/install_at_local.sh"
    conn.run(f"mkdir -p /YOUR_PATH", warn=True)
    conn.run(f"bash {script_path} {rank} {n_rank} {main_process_ip} {host} {project_name}", pty=True, warn=True)


@task
def check(conn, host, rank, n_rank):
    if host == local_ip:
        return
    script_path = "/YOUR_PATH/check.sh"
    run_docker_base(conn, host, rank, n_rank, script_path)
    conn.run(f'docker exec {docker_name} bash {script_path} {rank} {n_rank} {main_process_ip} {host}', pty=True)

@task
def install_new_machine(conn, host, rank, n_rank):
    if host == local_ip:
        return

    mount(conn, host)
    # script_path = "/root/mount.sh"
    # os.system(f"scp -r /home/save_dir/projects/yancen/SoraClusterScripts/mount.sh root@{host}:/root")
    # conn.run(f'bash {script_path}', pty=True, warn=True)

    # script_path = "/home/save_dir/projects/yancen/SoraClusterScripts/install_new_machine.sh"

    # conn.run(f"mkdir -p /home/save_dir/projects/yancen/SoraClusterScripts", warn=True)
    # conn.run(f"mkdir -p /home/opensora/captions/linbin_captions", warn=True)

    # print(f"{host}: copy /home/opensora/yancen/pre_weights...")
    # os.system(f"scp -r /home/opensora/yancen/pre_weights root@{host}:/home/opensora/yancen/")

    # print(f"{host}: copy /home/opensora/captions...")
    # os.system(f"scp -r /home/opensora/captions root@{host}:/home/opensora")

    # # print(f"{host}: copy /root/docker-19.03.10.tgz...")
    # # os.system(f"scp -r /root/docker-19.03.10.tgz root@{host}:/root/")

    # print(f"{host}: copy /home/opensora/sora_plan_v2.tar...")
    # os.system(f"scp -r /home/opensora/sora_plan_v2.tar root@{host}:/home/opensora/")

    # # 安装驱动和docker等
    # print(f"{host}: run {script_path}...")
    # conn.run(f"bash {script_path}")

    # # 安装docker
    # print(f"{host}: install_docker...")
    # install_docker(conn, host, rank, n_rank, load_docker=True)


def untar(conn, host, rank, n_rank):
    if host == local_ip:
        return
    script_path = f"{script_base_path}/untar.sh"
    conn.run(f"mkdir -p {script_base_path}", warn=True)
    conn.run(f"bash {script_path} {rank} {n_rank} {main_process_ip} {host}", pty=True, warn=True)


@task
def install_at_docker(conn, host, rank, n_rank):
    script_path = "/YOUR_PATH/install_at_docker.sh"
    project_name = args.project_name
    # assert project_name is not None
    # run_docker_base(conn, host, rank, n_rank, script_path)
    # conn.run("docker start ubuntu_zsh", warn=True)
    # conn.run(f'docker exec ubuntu_zsh zsh && python /home/save_dir/projects/yancen/SoraClusterScripts/resize_vid.py', pty=True, warn=True)
    conn.run(f'docker exec {docker_name} bash {script_path} {rank} {n_rank} {main_process_ip} {host} {project_name}', pty=True)


@task
def copy_sa_from_179(conn, host, rank, n_rank):
    # 获取所有tar文件
    tar_files = [f"sa_{i:06d}.tar" for i in range(601)]

    # 计算每台服务器分配的文件数量
    files_per_host = len(tar_files) // n_rank
    remainder = len(tar_files) % n_rank

    # 确定当前服务器分配的文件范围
    start_index = rank * files_per_host
    end_index = start_index + files_per_host
    if rank < remainder:
        start_index += rank
        end_index += rank + 1
    else:
        start_index += remainder
        end_index += remainder

    # 创建目标目录
    script_path = "/YOUR_PATH/copy_sa.sh"
    run_docker_base(conn, host, rank, n_rank, script_path)
    conn.run(f'docker exec {docker_name} bash {script_path} {start_index} {end_index - 1}', pty=True)


@task
def copy_video(conn, host, rank, n_rank):
    server_files_list = [
        # '/home/video_data/hzh/video_pexel_65f/server_files_1.json',
        # '/home/video_data/hzh/video_pexel_65f/server_files_0.json',
        '/home/video_data/hzh/video_pixabay_65f/server_files.json'
    ]
    # /home/video_data/video_pexel/,/home/opensora/captions/linbin_captions/video_pexel_65f_3832666.json
    # /home/video_data/myssd6/pixabay_v2/,/home/opensora/captions/linbin_captions/video_pixabay_65f_601513.json
    # /home/video_data/MJ/mixkit/mixkit/,/home/opensora/captions/linbin_captions/video_mixkit_65f_54735.json
    video_dir_list = [
        # '/home/video_data/MJ/mixkit/mixkit/',
        # '/home/video_data/video_pexel/',
        # '/home/video_data/myssd6/pixabay_v2/'
        # 'obs://sora/20240426/mydisk/',
        # 'obs://sora/20240426/mydisk2/',
        'obs://sora/20240426/myssd6/pixabay_v2',
    ]
    dest_dir_list = [
        # '/home/local_dataset/video_data_obs/pexel/',
        # '/home/local_dataset/video_data_obs/pexel/',
        '/home/local_dataset/video_data_obs/pixabay_v2/'
    ]

    # 自定义命令
    # conn.run(f"bash {script_base_path}/copy_obs.sh", pty=True, warn=True)

    for server_files, video_dir, dest_dir in zip(server_files_list, video_dir_list, dest_dir_list):
        with open(server_files, 'r') as f:
            server_files_data = json.load(f)

        for server_ip, files in server_files_data.items():
            if server_ip == host or (server_ip == "192.168.0.221" and host == "192.168.0.185"):
                n_file = len(files)
                for i, relative_path in enumerate(files):
                    video_abs_path = os.path.join(video_dir, relative_path)
                    dest_path = os.path.join(dest_dir, relative_path)
                    conn.run(f"mkdir -p {os.path.dirname(dest_path)}", warn=True)
                    cp_command = f"{obs_util} cp {video_abs_path} {dest_path}"
                    conn.run(cp_command, warn=True, pty=True)
                    print(f"[{host}]: progress: {i}/{n_file} of {video_dir}")

def run_copy(conn, host, rank, n_rank, sa_files, dest_dir):
    # 打乱文件顺序
    random.shuffle(sa_files)
    # 计算每台服务器分配的文件数量
    files_per_host = len(sa_files) // n_rank
    remainder = len(sa_files) % n_rank

    # 确定当前服务器分配的文件范围
    start_index = rank * files_per_host
    end_index = start_index + files_per_host
    if rank < remainder:
        start_index += rank
        end_index += rank + 1
    else:
        start_index += remainder
        end_index += remainder

    # 创建目标目录
    conn.run(f"rm -rf {dest_dir}")
    conn.run(f"mkdir -p {dest_dir}")

    # 拷贝文件到当前服务器
    n_file = end_index - start_index
    for i in range(start_index, end_index):
        src_file = sa_files[i]
        # 利用src_file和src_path算一个相对路径
        dest_file = f"{dest_dir}/{os.path.basename(src_file)}"
        cp_command = f"{obs_util} cp {src_file} {dest_file}"
        conn.run(cp_command)
        # print(f"Copied {src_file} to {host}:{dest_file}")
        print(f"[{host}]: progress: {i - start_index}/{n_file} of {src_file}")

@task
def copy_sa(conn, host, rank, n_rank):
    # OBS上SA文件所在的目录列表
    obs_dirs = [
        "obs://sora/20240426/raw/",
        "obs://sora/20240426/myssd3/dataset/sa-1b/",
        "obs://sora/20240426/myssd/",
        "obs://sora/20240426/bufa2/sa-bufa/",
        "obs://sora/20240426/bufa1/360-399/",
        "obs://sora/20240426/600-799/download_sam/",
        "obs://sora/20240426/0-199/dataset/sa-1b/",
        "obs://sora/20240426/800-999/download_sam_800_999/",
        "obs://sora/20240426/sa-1b/OpenDataLab___SA-1B/raw/"
    ]
    # obs://sora/20240426/raw/fa/test/1.tar /obs://sora/20240426/raw/fa
    def is_tar_file(f):
        if not f.startswith("obs"):
            return False
        base_name = os.path.basename(f)
        return base_name.startswith('sa_') and base_name.endswith('.tar')

    # 获取所有SA文件
    sa_files = []
    for obs_dir in obs_dirs:
        command = f"{obs_util} ls -s {obs_dir} -limit=100000"
        result = conn.run(command, warn=True)
        files = [f for f in result.stdout.split('\n') if is_tar_file(f)]
        sa_files.extend(files)
    #
    # return
    dest_dir = "/home/local_dataset/sa_files/"
    run_copy(conn, host, rank, n_rank, sa_files, dest_dir)


# load_docker=False, image_name="ubuntu_zsh:1.0.0", docker_name="ubuntu_zsh"

@task
def install_docker(conn, host, rank, n_rank, load_docker=True, image_name="hw:v1", docker_name="llama_test"):
    print(f"Start process {host}: copy docker file...")
    result = conn.run(f'docker inspect --format="{{.State.Running}}" {docker_name}', warn=True)
    # conn.run("mkdir /home/opensora", warn=True)

    if load_docker:
        # os.system(f"scp -r /home/opensora/sora_plan_v2.tar root@{host}:/home/opensora/")
        print(f"{host}: load docker file and install docker...")
        conn.run(f"docker load < /home/save_dir/llama_test/hw.tar")
        print(f"{host}: run docker command...")

    # conn.run("systemctl start docker", warn=True) 
    # conn.run("sleep 2s", warn=True)
    print(f"{host}: detect {docker_name}, if find, remove it")
    conn.run(f"docker stop {docker_name}", warn=True)
    conn.run(f"docker rm {docker_name}", warn=True)
    conn.run(f"docker run -itd --network host --privileged --name={docker_name} --ipc=host -u root "
             "--device=/dev/davinci0 "
             "--device=/dev/davinci1 "
             "--device=/dev/davinci2 "
             "--device=/dev/davinci3 "
             "--device=/dev/davinci4 "
             "--device=/dev/davinci5 "
             "--device=/dev/davinci6 "
             "--device=/dev/davinci7 "
             "--device=/dev/davinci_manager "
             "--device=/dev/devmm_svm "
             "--device=/dev/hisi_hdc "
             "-v /usr/slog:/usr/slog "
             "-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi "
             "-v /etc/hccn.conf:/etc/hccn.conf "
             "-v /usr/local/Ascend/driver:/usr/local/Ascend/driver "
             "-v /usr/slog:/usr/slog "
             "-v /etc/ascend_install.info:/etc/ascend_install.info "
             "-v /home/obs_data:/home/obs_data "
             "-v /home/save_dir:/home/save_dir "
             f"{image_name} /bin/bash", pty=True)


def main():
    if args.command == "setup_ssh_key":
        # 设置其他机器的免密登录
        setup_ssh_key_for_local()
        # 设置其它机器登录本机器免密
        for rank, host in enumerate(hosts):
            conn = Connection(f"root@{host}", connect_kwargs={'password': password})
            setup_ssh_key_for_others(conn)
        return

    if args.command == "untar.py":
        host_file_lists = get_host_file_lists()

    n_rank = len(hosts)
    if args.parallel:
        threads = []
        for rank, host in enumerate(hosts):
            conn = Connection(f"root@{host}", connect_kwargs={'password': password})
            print(f"{'=' * 60} Processing of host {host} {'=' * 60}")

            if args.command == "untar.py":
                thread = threading.Thread(target=map_command2fcn[args.command],
                                          args=(conn, host, rank, n_rank, host_file_lists[rank]))
            else:
                thread = threading.Thread(target=map_command2fcn[args.command], args=(conn, host, rank, n_rank))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
    else:
        n_rank = len(hosts)
        for rank, host in enumerate(hosts):
            print(f"{'=' * 60} Processing of host {host} {'=' * 60}")
            conn = Connection(f"root@{host}", connect_kwargs={'password': password})
            map_command2fcn[args.command](conn, host, rank, n_rank)


if __name__ == "__main__":
    map_command2fcn = {"install_docker": install_docker,
                       "install_at_local.sh": install_at_local,
                       "install_at_docker.sh": install_at_docker,
                       "install_new_machine.sh": install_new_machine,
                       "tail_log.sh": print_latest_log,
                       "train.sh": train,
                       "test.sh": test,
                       "check.sh": check,
                       "test_connection.sh": test_connection,
                       "kill.sh": kill,
                       "reboot_npu.sh": reboot_npu,
                       "setup_ssh_key": "setup_ssh_key",
                       "untar.sh": untar,
                       "copy_video": copy_video,
                       "copy_sa": copy_sa,
                       "copy_sa_from_179.sh": copy_sa_from_179,
                       }

    parser = argparse.ArgumentParser(description="Execute commands in serial or parallel mode.")
    parser.add_argument("-c", "--command", type=str, help=f"The command to execute")
    # parser.add_argument("-ip", "--ip_list", default="ip_lists/ip_list.txt", type=str, help=f"ip_list_file")
    parser.add_argument("-ip", type=int, nargs=2, default=[0, 48], help="Enter start and end values as --range start end")
    parser.add_argument("-p", "--parallel", action="store_true", help="Run the command in parallel mode.")
    parser.add_argument("-pn", "--project_name", type=str, help="Project name", default=None)
    parser.add_argument("-ps", "--project_script", type=str, help="Project name", default=None)
    parser.add_argument("-nf", "--num_frame", type=int, help="Project name", default=None)
    parser.add_argument("-wt", "--width", type=int, help="Project name", default=None)
    parser.add_argument("-ht", "--height", type=int, help="Project name", default=None)

    args = parser.parse_args()
    assert args.command in map_command2fcn.keys()

    # 读取IP列表文件
    with open("/YOUR_PATH/ip_test.txt", 'r') as f:
        hosts = [line.strip() for line in f.readlines()]

    hosts = hosts[args.ip[0]: args.ip[1]]
    # 设置统一的密码
    password = 'Fy!mATB@QE'
    # 本地ip
    local_ip = os.popen('hostname -I').read().strip().split()[0]
    docker_name = "llama_test"

    # 主节点
    main_process_ip = hosts[0]
    # base_path = "/home/save_dir/projects/yancen/Open-Sora-Plan"
    base_path = "/YOUR_PATH"
    script_base_path = "/YOUR_PATH"
    obs_util = "/YOUR_PATH/obsutil"

    main()
