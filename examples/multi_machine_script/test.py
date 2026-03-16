import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

torch_npu.npu.set_compile_mode(jit_compile=False)
import torch.distributed as dist

from datetime import timedelta
import time

# 设置超时时间为 30 秒
timeout = timedelta(seconds=30)


def run(rank, size):
    tensor = torch.ones(1).to(rank % 8)
    start_time = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end_time = time.time()
    print(f"[rank]: Rank {rank} has data {tensor[0]} on GPU {rank % 8}, time={end_time - start_time}", flush=True)


def main():
    rank = int(os.environ['RANK'])
    torch_npu.npu.set_device(rank % 8)
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size,
                            init_method=f'tcp://{master_addr}:{master_port}',timeout=timeout)
    run(rank, world_size)


if __name__ == "__main__":
    main()
