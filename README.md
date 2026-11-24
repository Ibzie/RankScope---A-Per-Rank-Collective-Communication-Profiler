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

## Hardware notes

Everything up to the two-node link runs on a single GPU. Real multi-rank
collectives need a second machine, because NCCL requires one distinct device
per rank. See PLAN.md §2 and §5 (Step 0.5).
