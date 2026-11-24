#!/usr/bin/env python3
"""Can we run real multi-rank NCCL on a single GPU?

NCCL 2.30 added multi-rank-per-GPU support behind NCCL_MULTI_RANK_GPU_ENABLE=1.
This probe answers the questions that decide how much of this project can be
built on one machine:

  1. Does it actually work, and is the result *correct*?
  2. How many ranks can share cuda:0 before it breaks?
  3. What does a collective cost at each world size, and where does it stop
     being latency-bound and start being bandwidth-bound?

Correctness is checked, not assumed: each rank contributes rank+1, so the
reduced value must be world_size * (world_size + 1) / 2 on every rank. A
transport bug that silently drops a contribution shows up here.

Run:  uv run labs/00b_multirank_probe.py
      uv run labs/00b_multirank_probe.py --world-sizes 2 3 4
"""

from __future__ import annotations

import argparse
import os
import socket
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES_MIB = (0.004, 0.016, 0.064, 0.256, 1, 4, 16)
ITERS = 50


def _worker(rank: int, world: int, port: int) -> None:
    try:
        torch.cuda.set_device(0)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world,
            # A wrong collective sequence deadlocks rather than erroring, so
            # give it a deadline: a traceback beats an infinite wait.
            timeout=timedelta(seconds=90),
        )

        _check_correct(rank, world)
        _measure(rank, world)

        dist.barrier()
        dist.destroy_process_group()

    except Exception:
        print(f"[rank {rank}] {traceback.format_exc()}", flush=True)
        raise


def _check_correct(rank: int, world: int) -> None:
    expected = world * (world + 1) / 2  # sum of 1..world
    x = torch.full((4096,), float(rank + 1), device="cuda")
    dist.all_reduce(x)
    ok = torch.allclose(x, torch.full_like(x, expected), rtol=0, atol=1e-6)

    # Every rank must agree, so each one reports rather than trusting rank 0.
    flag = torch.tensor([1.0 if ok else 0.0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        got = x[0].item()
        status = "ok" if flag.item() == 1 else "MISMATCH"
        print(f"  world={world:<2} correctness {status:<9} (expect {expected:g}, got {got:g})", flush=True)


def _measure(rank: int, world: int) -> None:
    """Every rank must run this loop, because a collective is a joint operation.

    Only rank 0 records and reports: the other ranks would report their own
    view, which is the same number but three copies of it.
    """
    if rank == 0:
        print(f"  {'size':>8}  {'latency':>11}  {'algo BW':>10}  {'bus BW':>10}")
    for size_mib in SIZES_MIB:
        n = int(size_mib * 2**20) // 4
        buf = torch.ones(n, dtype=torch.float32, device="cuda")

        for _ in range(10):
            dist.all_reduce(buf)
        torch.cuda.synchronize()

        # Time a batch, not one op. A training loop issues collectives
        # asynchronously and they pipeline; timing one op at a time forces a
        # full drain, which measures our synchronisation, not NCCL.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            dist.all_reduce(buf)
        end.record()
        torch.cuda.synchronize()

        ms = start.elapsed_time(end) / ITERS

        if rank == 0:
            nbytes = n * 4
            algbw = nbytes / (ms / 1e3) / 1e9
            # bus BW = what each link actually carried: 2*(N-1)/N of the buffer
            busbw = algbw * 2 * (world - 1) / world

            label = f"{size_mib * 1024:.0f}K" if size_mib < 1 else f"{size_mib:.0f}M"
            print(
                f"  {label:>8}  {ms * 1e3:8.1f} us  {algbw:7.2f} GB/s  {busbw:7.2f} GB/s",
                flush=True,
            )

        del buf
        torch.cuda.empty_cache()


def _free_port() -> int:
    """Ask the kernel for an unused port.

    Hard-coding port numbers means a crashed run poisons the next one with
    EADDRINUSE, which looks like a rendezvous bug rather than a leftover socket.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def probe(world: int) -> bool:
    port = _free_port()
    print(f"\nworld_size={world} on cuda:0")
    try:
        mp.spawn(_worker, args=(world, port), nprocs=world, join=True)
        return True
    except Exception as exc:  # spawn re-raises the child's failure
        print(f"  FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-sizes", type=int, nargs="+", default=[2, 4, 8])
    args = ap.parse_args()

    # The whole probe depends on this opt-in.
    os.environ["NCCL_MULTI_RANK_GPU_ENABLE"] = "1"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    mp.set_start_method("spawn", force=True)

    ok = [w for w in args.world_sizes if probe(w)]
    print(f"\nworld sizes that worked on one GPU: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
