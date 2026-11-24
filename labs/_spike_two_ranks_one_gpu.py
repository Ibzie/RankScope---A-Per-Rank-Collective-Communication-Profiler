#!/usr/bin/env python3
"""Deliberately attempt two NCCL ranks on ONE physical GPU.

Expected outcome: NCCL aborts with "Duplicate GPU detected". That is not a bug
in this file -- it is the constraint that decides the architecture of this
project, and we want to see it rather than take it on faith.

NCCL treats this as a fatal error and calls abort(), so this script must be run
as a *subprocess* by the main lab. Otherwise the abort takes the whole lab with
it.

On a genuine multi-GPU box, the only line that would change is
`torch.cuda.set_device(rank)`.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 2
PORT = 29511


def worker(rank: int) -> None:
    # Force both ranks onto device 0. This is the whole experiment.
    torch.cuda.set_device(0)

    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{PORT}",
        rank=rank,
        world_size=WORLD_SIZE,
    )

    t = torch.arange(1024, dtype=torch.float32, device="cuda")
    dist.all_reduce(t)
    dist.barrier()

    if rank == 0:
        print("RESULT: two ranks initialized on cuda:0 and completed all_reduce", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(), nprocs=WORLD_SIZE, join=True)
