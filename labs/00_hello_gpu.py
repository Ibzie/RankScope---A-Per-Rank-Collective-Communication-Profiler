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


def probe_two_ranks_one_gpu(timeout_s: int = 180) -> tuple[bool, str, str]:
    """Run the spike in a subprocess. Returns (constraint_held, verdict, detail).

    NCCL aborts on duplicate devices, so we must not do this in-process.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(HERE / "_spike_two_ranks_one_gpu.py")],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", "the job hung instead of aborting (also a finding)"

    combined = proc.stdout + proc.stderr

    if "Duplicate GPU" in combined:
        detail = next(
            (ln.strip() for ln in combined.splitlines() if "Duplicate GPU" in ln), ""
        )
        return True, "FAILED AS EXPECTED", detail

    if proc.returncode != 0:
        return False, "FAILED, WRONG REASON", tail(combined)

    return False, "UNEXPECTEDLY SUCCEEDED", tail(combined)


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
    print("  'algo BW' = bytes / time. With one rank there is no bus to")
    print("  divide by, so this is a floor, not a bandwidth.")

    section("5. can two ranks share one GPU?")
    held, verdict, detail = probe_two_ranks_one_gpu()
    row("verdict", verdict)
    if detail:
        for line in detail.splitlines():
            row("", line)

    print()
    if held:
        print("  Confirmed: one device per rank. Real collectives need the")
        print("  second machine (PLAN.md, Step 0.5).")
    else:
        print("  The constraint did NOT hold as expected. Record what")
        print("  actually happened before trusting the rest of the plan.")

    return 0 if held else 2


if __name__ == "__main__":
    raise SystemExit(main())
