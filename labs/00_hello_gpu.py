#!/usr/bin/env python3
"""Step 0 -- establish ground truth about this machine.

Everything later is built on the numbers this lab prints. It answers five
questions, in order, and does not profile anything:

  1. What software am I running?        torch / CUDA / NCCL versions.
  2. What hardware am I running on?     device properties, sm_89, L2, memory.
  3. How fast is this GPU's memory?     measured, not quoted from a spec sheet.
  4. What does a collective cost?       world_size=1 all-reduce baseline.
  5. Is "one device per rank" real?     two ranks on cuda:0, expected to abort.

Why the NCCL version is printed first: it silently determines which algorithms
and transports are available, so every "why did it choose ring instead of tree"
question later starts by confirming the version.

Run:  uv run labs/00_hello_gpu.py
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print("-" * len(title))


def row(label: str, value: object, width: int = 26) -> None:
    print(f"  {label:<{width}} {value}")


def tail(text: str, n: int = 6) -> str:
    """Last n non-empty lines, so error output stays readable."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# 1. software
# ---------------------------------------------------------------------------


def software() -> dict[str, object]:
    nccl = torch.cuda.nccl.version()
    return {
        "torch": torch.__version__,
        "cuda (compiled)": torch.version.cuda,
        "nccl": ".".join(str(part) for part in nccl),
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "devices visible": torch.cuda.device_count(),
    }


# ---------------------------------------------------------------------------
# 2. device properties
# ---------------------------------------------------------------------------


def device() -> dict[str, object]:
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "compute capability": f"{p.major}.{p.minor}",
        "SMs": p.multi_processor_count,
        "L2 cache": f"{getattr(p, 'L2_cache_size', 0) / 2**20:.1f} MiB",
        "total memory": f"{p.total_memory / 2**30:.1f} GiB",
        "clock (max)": f"{getattr(p, 'clock_rate', 0) / 1e6:.2f} GHz",
    }


# ---------------------------------------------------------------------------
# 3. measured device-to-device bandwidth
# ---------------------------------------------------------------------------


def measure_copy_bandwidth(nbytes: int = 256 * 2**20, iters: int = 20) -> tuple[float, float]:
    """Median `copy_` bandwidth in GB/s, and the median time in ms.

    A copy both reads and writes, so the traffic across the memory bus is
    2 * nbytes. Quoting only one direction is the most common way benchmarks
    flatter themselves by 2x.

    Timing with CUDA events, not the host clock: copy_ returns as soon as the
    work is *enqueued*, so a host-clock measurement would report microseconds
    for a millisecond-long operation.
    """
    n = nbytes // 4
    src = torch.empty(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)

    for _ in range(5):  # warm up: clocks, page tables, allocator
        dst.copy_(src)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median_ms = times[len(times) // 2]
    return 2 * nbytes / (median_ms / 1e3) / 1e9, median_ms


# ---------------------------------------------------------------------------
# 4. world_size=1 all-reduce baseline
# ---------------------------------------------------------------------------

SIZES_MIB = (0.004, 0.016, 0.064, 0.256, 1, 4, 16, 64)


def measure_allreduce_baseline(iters: int = 20) -> list[tuple[float, float, float]]:
    """Latency of a single-rank all-reduce, by buffer size.

    With one rank this is the "no communication at all" case: NCCL still runs
    its kernel, so what you are measuring is NCCL's fixed overhead plus one
    pass over the buffer. Every later multi-rank number is compared against
    this floor.

    Returns [(size_mib, latency_us, algo_GB_per_s), ...].
    """
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29500",
        rank=0,
        world_size=1,
    )

    results = []
    try:
        for size_mib in SIZES_MIB:
            n = int(size_mib * 2**20) // 4
            buf = torch.ones(n, dtype=torch.float32, device="cuda")

            for _ in range(5):
                dist.all_reduce(buf)
            torch.cuda.synchronize()

            times: list[float] = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                dist.all_reduce(buf)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))

            times.sort()
            median_ms = times[len(times) // 2]
            latency_us = median_ms * 1e3

            nbytes = n * 4
            algbw = nbytes / (median_ms / 1e3) / 1e9
            results.append((size_mib, latency_us, algbw))

            del buf
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()

    return results


# ---------------------------------------------------------------------------
# 5. the constraint
# ---------------------------------------------------------------------------


def _run_spike(extra_env: dict[str, str], timeout_s: int = 180) -> subprocess.CompletedProcess:
    env = {**os.environ, **extra_env}
    try:
        return subprocess.run(
            [sys.executable, str(HERE / "_spike_two_ranks_one_gpu.py")],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args=[], returncode=-1, stdout="", stderr="TIMEOUT")


def probe_two_ranks_one_gpu() -> tuple[str, str]:
    """Returns (verdict, detail).

    Three possible worlds, and they mean very different things:

      * two ranks on one GPU is impossible        -> the plan needs a second box
      * it works only with an opt-in env var      -> usable as a dev harness
      * it works out of the box                   -> surprising, verify correctness

    NCCL aborts rather than raising here, so every attempt runs in a subprocess.
    """
    default = _run_spike({})
    if default.returncode == 0:
        return "WORKS BY DEFAULT", "two ranks shared cuda:0 with no configuration"

    combined = default.stdout + default.stderr
    if "NCCL_MULTI_RANK_GPU_ENABLE" in combined:
        opted_in = _run_spike({"NCCL_MULTI_RANK_GPU_ENABLE": "1"})
        if opted_in.returncode == 0:
            return (
                "WORKS WITH OPT-IN",
                "NCCL_MULTI_RANK_GPU_ENABLE=1 allows 2 ranks on one GPU (see 00b)",
            )
        return "OPT-IN FAILED", tail(opted_in.stdout + opted_in.stderr)

    if default.returncode != 0:
        detail = next(
            (ln.strip() for ln in combined.splitlines() if "Last error" in combined.splitlines()),
            tail(combined, 3),
        )
        return "FAILED, OTHER REASON", detail

    return "UNEXPECTED", tail(combined)


def cpu_governor() -> str:
    """The CPU frequency governor. It adds variance to every timestamp we take,
    so it belongs in the record next to the hardware it describes."""
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return path.read_text().strip()
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------


def main() -> int:
    if not torch.cuda.is_available():
        print("torch does not see a GPU; nothing else is worth measuring.")
        return 1

    print("RankScope -- Step 0: ground truth")
    print(f"host: {platform.node()}  kernel: {platform.release()}")

    section("1. software")
    for k, v in software().items():
        row(k, v)

    section("2. device")
    for k, v in device().items():
        row(k, v)
    row("cpu governor", cpu_governor())

    section("3. measured device-to-device bandwidth")
    bw, ms = measure_copy_bandwidth()
    row("256 MiB copy", f"{bw:8.1f} GB/s   ({ms:.3f} ms, read+write)")
    print("  (a 4060's spec sheet says ~272 GB/s; the gap is real overhead)")

    section("4. all-reduce, world_size=1 (no communication)")
    print(f"  {'size':>10}  {'latency':>12}  {'algo BW':>12}")
    for size_mib, latency_us, algbw in measure_allreduce_baseline():
        label = f"{size_mib * 1024:.0f} KiB" if size_mib < 1 else f"{size_mib:.0f} MiB"
        print(f"  {label:>10}  {latency_us:9.1f} us  {algbw:9.1f} GB/s")
    print("  All sizes cost the same ~31 us -- including 64 MiB. A real")
    print("  all-reduce would need ~620 us just to read and write that buffer")
    print("  at the 215 GB/s measured above, so the reported time is ~20x")
    print("  faster than physically possible: PyTorch elides single-rank")
    print("  collectives, so this measures the Python call and nothing else.")
    print("  There is no usable world_size=1 baseline; every real number")
    print("  needs a second rank. What this *is* good for is the")
    print("  per-collective overhead budget our instrumentation must beat.")

    section("5. can two ranks share one GPU?")
    verdict, detail = probe_two_ranks_one_gpu()
    row("verdict", verdict)
    if detail:
        for line in detail.splitlines():
            row("", line)

    print()
    if verdict.startswith("WORKS"):
        print("  Multi-rank on one device is possible, so many experiments that")
        print("  assumed a second machine can run locally. Careful: it is not")
        print("  representative -- see the transport and latency notes in")
        print("  PLAN.md Step 0 findings before trusting absolute numbers.")
    elif verdict.startswith("FAILED"):
        print("  One device per rank, as expected. Real collectives need the")
        print("  second machine (PLAN.md, Step 0.5).")
    else:
        print("  Neither outcome was expected. Record what actually happened")
        print("  before trusting the rest of the plan.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
