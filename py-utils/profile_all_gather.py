"""
launch: 
python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 py-utils/profile_all_gather.py
"""

import torch 
import numpy as np
from torch import distributed as dist
import time
import matplotlib.pyplot as plt

def print_at_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)

def benchmark_all_gather(n, warm_up=5, repeat=10):
    world_size = dist.get_world_size()
    device_id = dist.get_rank() % torch.cuda.device_count()

    local_tensor = torch.rand((n,), device=f'cuda:{device_id}', dtype=torch.float)
    tensor_list = [torch.zeros(n, dtype=torch.float, device=f'cuda:{device_id}') for _ in range(world_size)]

    ts = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(warm_up+repeat):
        t1 = time.time()
        start_event.record()
        dist.all_gather(tensor_list, local_tensor)
        end_event.record()
        end_event.synchronize()

        if i >= warm_up:
            ts.append(start_event.elapsed_time(end_event))

    avg_t = np.mean(ts)
    bw = (world_size-1) * n * 4 / 1e9 / (avg_t/1e3)
    psize = n * 4 / 1e6

    print_at_rank0(f'partition size {n*4/1e6:.5} MB, average time {np.mean(ts):.4} ms, bw {bw:.4} GB/s')
    return avg_t, bw, psize

def main():
    dist.init_process_group(backend='nccl')
    local_size = torch.cuda.device_count()
    rank = dist.get_rank()
    torch.cuda.set_device(rank % local_size)
    # print(f'rank {rank}')
    
    nelem_range = np.concatenate(
        (
            np.arange(100, 1e3, 100), 
            np.arange(1e3, 1e4, 1e3),
            np.arange(1e4, 1e5, 1e4),
            np.arange(1e5, 1e6, 1e5),
            np.arange(1e6, 1e7, 1e6),
            np.arange(1e7, 4e7, 1e7)
        ))
    
    rs = []
    for n in nelem_range:
        n = int(n)
        r = benchmark_all_gather(n)
        rs.append(r)
    rs = np.array(rs)
    
    plt.plot(rs[:, -1], rs[:, -2])
    plt.xlabel('all_gather partition size (MB)')
    plt.ylabel('bandwidth (GB/s)')
    plt.title('pytorch.distributed.all_gather profile')
    plt.savefig('all_gather_profile.png')

if __name__ == "__main__":
    main()