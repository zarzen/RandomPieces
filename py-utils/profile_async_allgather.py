"""
launch: 
python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 py-utils/profile_all_gather.py
"""

import torch 
import numpy as np
from torch import distributed as dist
from torch.cuda import nvtx
import time

from torch.distributed.distributed_c10d import all_gather, _batch_p2p_manager, get_backend
from torch.distributed.distributed_c10d import _get_default_group, _pg_names


def print_at_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)



def benchmark_all_gather(partition_sizes, local_params, comm_stream):
    dtype = torch.half
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    t1 = time.time()
    with torch.cuda.stream(comm_stream):
        nvtx.range_push('allocate final params')
        # allocate memories
        allgather_params = []
        for psize in partition_sizes:
            tensor_size = psize * world_size
            tensor = torch.empty(tensor_size, dtype=dtype, device=f'cuda:{device_id}').view(-1)
            allgather_params.append(tensor)

        nvtx.range_pop()
    comm_stream.synchronize()
    t2 = time.time()
    # print_at_rank0(f'allocate cost {t2 - t1} s')

    with torch.cuda.stream(comm_stream):
        nvtx.range_push('construct all output list')
        # create allgather parameters
        all_gather_list_list = []
        for pidx, psize in enumerate(partition_sizes):
            flat_tensor = allgather_params[pidx]
            partitions = []
            for i in range(world_size):
                partitions.append(flat_tensor.narrow(0, psize * i, psize))

            all_gather_list_list.append(partitions)

        nvtx.range_pop()

    comm_stream.synchronize()
    print_at_rank0(f'construct params cost {time.time() - t2} s')

    with torch.cuda.stream(comm_stream):
        backend = get_backend()
        nvtx.range_push('launch dist all-gather')

        with _batch_p2p_manager(backend):
            handles = []    
            for pidx, psize in enumerate(partition_sizes):
                h = all_gather(all_gather_list_list[pidx], 
                                all_gather_list_list[pidx][rank], 
                                async_op=True)
                # h = dist.all_gather(all_gather_list_list[pidx], 
                #                 all_gather_list_list[pidx][rank], 
                #                 async_op=True)

                handles.append(h)
        
        # handles=[]
        # for pidx, psize in enumerate(partition_sizes):
        #     # h = all_gather(all_gather_list_list[pidx], 
        #     #                 all_gather_list_list[pidx][rank], 
        #     #                 async_op=True)
        #     h = dist.all_gather(all_gather_list_list[pidx], 
        #                     local_params[pidx], 
        #                     async_op=True)
        #     handles.append(h)
        #     # torch.cuda.synchronize()

        # handles[-1].wait() # event enqueued, but not guaranteed complete
        nvtx.range_pop()

    torch.cuda.synchronize()
    end_event = torch.cuda.Event()
    comm_stream.wait_event(end_event)
    return None

def main():
    # c10d_frontend = torch.classes.dist_c10d.frontend()

    dist.init_process_group(backend='nccl')
    d_pg = _get_default_group()
    # print(c10d_frontend.get_name_of_process_group(d_pg))

    pg2 = dist.new_group([0,1], backend='nccl')
    # print(c10d_frontend.get_name_of_process_group(pg2))
    if dist.get_rank() == 0:
        print(dir(d_pg))
        print(type(d_pg))
        print(type(pg2))
        print(dir(dist))
        print(_pg_names)

    local_size = torch.cuda.device_count()
    rank = dist.get_rank()
    torch.cuda.set_device(rank % local_size)
    torch.cuda.synchronize()
    comm_stream = torch.cuda.Stream(rank % local_size)
    device_id = rank % local_size
    # print(f'rank {rank}')
    warm_up = 5
    repeat = 10

    partition_sizes = [
        2457600,
        960,
        819200,
        320,
        320,
        320,
        3276800,
        1280,
        3276800,
        320,
        320,
        320
    ]

    local_params = []
    for psize in partition_sizes:
        r = torch.rand(psize, dtype=torch.half, device=f'cuda:{device_id}').view(-1)
        local_params.append(r)
        print(f'rank {rank}, psize {psize}, sum {torch.sum(r).item()}')

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True) 
    ts = []
    for i in range(repeat + warm_up):
        with torch.cuda.stream(comm_stream):
            nvtx.range_push(f'exp-{i}')
            t1 = time.time()
            # start_event.record(stream=comm_stream)
            benchmark_all_gather(partition_sizes, local_params, comm_stream)
            # end_event.record(stream=comm_stream)
            # end_event.synchronize()

            t2 = time.time()
            nvtx.range_pop()

            if i >= warm_up:
                # ts.append(start_event.elapsed_time(end_event))
                ts.append((t2 - t1) * 1e3)
    
    if dist.get_rank() == 0:
        avg_t = np.mean(ts)
        bw = (dist.get_world_size() - 1) * np.sum(partition_sizes) * 2 / 1e9 / (avg_t / 1e3)
        print(f'avg time {avg_t} ms, bw {bw} GB/s')


if __name__ == "__main__":
    main()