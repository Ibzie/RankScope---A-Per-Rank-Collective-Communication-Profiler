# RankScope

Per-rank collective communication profiler.

Finds the rank that is slowing down your all-reduce, and says whether it is
**late to arrive** (compute/CPU-slow) or **late to leave** (network-slow).

The plan lives in [PLAN.md](PLAN.md). The original design document is archived
at [docs/nccl-profiler-design-doc.md](docs/nccl-profiler-design-doc.md).

## Setup

```bash
uv sync            # creates .venv with Python 3.12 + torch 2.14.1+cu130
```

Verify:

```bash
uv run labs/00_hello_gpu.py
```

## Layout

```
src/rankscope/      the library (grows from Step 1 onward)
labs/               standalone experiments, one question each
docs/               design notes, archived design doc, measurement notes
```

## Measured baseline (RTX 4060, torch 2.14.1+cu130, NCCL 2.30.7)

| | |
|---|---|
| `copy_` 256 MiB, read+write | 215 GB/s |
| `all_reduce` world\_size=1 | elided by PyTorch — 31 µs regardless of size, so not a baseline |
| 2 ranks on 1 GPU (`NCCL_MULTI_RANK_GPU_ENABLE=1`) | correct, but **4.6 ms** fixed cost per collective |
| 4 ranks on 1 GPU | correct, but **27.6 ms** fixed cost per collective |

The single-GPU harness is for developing the profiler, not for measuring NCCL.
See PLAN.md §5 (Step 0 findings).

## Hardware notes

Everything up to the two-node link runs on a single GPU. Real multi-rank
collectives over a network need a second machine, because NCCL's `NET` transport
requires ibverbs (InfiniBand or RoCE) and has no TCP data path. See PLAN.md §2
and §5 (Step 0.5).
