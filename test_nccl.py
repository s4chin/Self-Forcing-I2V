import os
import torch
import torch.distributed as dist

def main():
    # 1. Initialize the distributed environment
    dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 2. Assign the current process to the correct local GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 3. Create a dummy tensor (simulate your random_seed)
    # Rank 0 sets it to 42, all other ranks set it to 0
    if global_rank == 0:
        seed_tensor = torch.tensor([42], dtype=torch.long, device=device)
    else:
        seed_tensor = torch.tensor([0], dtype=torch.long, device=device)

    print(f"[Rank {global_rank}] Before broadcast: {seed_tensor.item()}")

    # 4. The moment of truth: broadcast from Rank 0 to all other Ranks
    dist.broadcast(seed_tensor, src=0)
    
    # 5. Wait for all GPUs to catch up
    dist.barrier()

    print(f"[Rank {global_rank}] After broadcast: {seed_tensor.item()} - SUCCESS!")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()