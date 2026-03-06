"""Minimal NCCL multi-node test — reproduces the broadcast error without any
model/data downloads.  Run via torchrun (see nccl_test.sh)."""

import os
import socket
import torch
import torch.distributed as dist
from datetime import timedelta

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:
        init_method = f"tcp://[{host}]:{port}"
    else:
        init_method = f"tcp://{host}:{port}"

    print(f"[rank {rank}] hostname={socket.gethostname()} "
          f"init_method={init_method} world_size={world_size} local_rank={local_rank}")

    dist.init_process_group(
        rank=rank, world_size=world_size, backend="nccl",
        init_method=init_method, timeout=timedelta(minutes=10),
    )
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    print(f"[rank {rank}] process group initialized, device={device}")

    # This is the exact operation that fails in distillation.py:49
    tensor = torch.randint(0, 10000000, (1,), device=device)
    print(f"[rank {rank}] pre-broadcast value={tensor.item()}")

    dist.broadcast(tensor, src=0)

    print(f"[rank {rank}] post-broadcast value={tensor.item()} — NCCL OK")

    dist.barrier()
    print(f"[rank {rank}] barrier passed — all collective ops working")

    dist.destroy_process_group()
    print(f"[rank {rank}] done")


if __name__ == "__main__":
    main()
