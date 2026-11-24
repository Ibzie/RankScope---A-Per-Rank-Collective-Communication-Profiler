# RankScope — A Per-Rank Collective Communication Profiler

*Design document & implementation plan*

## 1. Problem Statement

Distributed training throughput is silently degraded by slow collective
operations. An all-reduce is a synchronization barrier: no rank proceeds until
every rank contributes, so the effective speed of the cluster is the speed of
its *slowest link at each step*. Two failure modes result:

- **Stragglers** — one rank is persistently slow (degraded NIC, thermal
  throttling, bad cable, misconfigured route). The whole cluster waits on it,
  every step, for hours, and nothing in standard monitoring points at it.
- **Jitter** — no rank is consistently slow, but some rank is randomly slow at
  random times (OS noise, checkpoint writes, data loader hiccups, network
  bursts). Invisible per-event, enormous in aggregate.

The raw data to diagnose this *exists* — every rank knows when it entered and
exited each collective — but it is scattered across thousands of processes and
never aggregated into a human-readable view. GPU utilization dashboards
actively mislead: a stalled all-reduce often still shows the GPU as busy.

**Goal:** continuous, near-zero-overhead, per-rank timing of collectives,
aggregated into a live view that answers three questions:

1. *Which* rank(s) slowed down this collective?
2. *Why class* of problem — compute-slow (arrived late) or network-slow
   (arrived on time, exited late)?
3. *Is it persistent* (straggler) or *transient* (jitter), and is it
   topologically correlated (same switch / node)?

## 2. Background Primer

(Short version — enough to make the design decisions legible.)

- **Data parallelism**: every GPU holds a full model copy, trains on a
  different data slice, and must combine gradients each step via **all-reduce**
  so the copies stay identical.
- **All-reduce algorithm** (ring): each rank splits its buffer into N chunks,
  passes them around a logical ring in a reduce-scatter phase then an
  all-gather phase. Cost per rank is ~constant as you scale — but every hop is
  a sync point, so the ring moves at the speed of its slowest link.
- **Other collectives**: broadcast, reduce, all-gather, reduce-scatter.
  Tensor parallelism and FSDP/ZeRO use these *inside* layers, making
  communication an even larger share of step time.
- **NCCL** implements the collectives, auto-selecting ring/tree algorithms and
  transports (NVLink intra-node, InfiniBand/RoCE inter-node). Its automatic
  choices are frequently suboptimal for a given cluster, and its observability
  story is debug logs and a per-process flight recorder — raw material, no
  aggregation layer.

## 3. Goals and Non-Goals

### Goals

- Zero code changes to the user's training script beyond one import/wrapper.
- < 1% runtime overhead; the profiler must never become the straggler.
- Works with PyTorch DDP and FSDP out of the box.
- Live dashboard: rank × time heatmap, per-collective breakdown, alerts.
- Offline mode: record to disk, analyze later (like a flight recorder).
- Explicit learning value: each phase teaches a real systems domain.

### Non-Goals

- Not a general training profiler (no operator-level tracing — PyTorch
  profiler already does that).
- Not a fixer: it diagnoses and attributes, it does not retune NCCL.
- Not NCCL-version-specific in phase 1 (pure PyTorch-level hooks first).

## 4. Architecture Overview

```
┌─────────────┐   ┌─────────────┐        ┌─────────────┐
│  Rank 0     │   │  Rank 1     │  ...   │  Rank N     │
│ ┌─────────┐ │   │ ┌─────────┐ │        │ ┌─────────┐ │
│ │  Hook   │ │   │ │  Hook   │ │        │ │  Hook   │ │
│ │ (timing)│ │   │ │ (timing)│ │        │ │ (timing)│ │
│ └────┬────┘ │   │ └────┬────┘ │        │ └────┬────┘ │
│      │ ring buffer     │               │      │
└──────┼───────┘   └──────┼──────┘        └──────┼──────┘
       │  batched push (gRPC / UDP / file sink)  │
       └──────────────┬───┴──────────────────────┘
                      ▼
              ┌───────────────┐
              │  Collector    │  ← clock-offset estimation lives here
              │  (aggregator) │
              └──────┬────────┘
                     ▼
              ┌───────────────┐
              │  Analysis     │  skew attribution, correlation, alerts
              └──────┬────────┘
                     ▼
              ┌───────────────┐
              │  Dashboard    │  heatmap, drill-down, alert feed
              └───────────────┘
```

Design principles:

1. **Collect at the source, judge at the center.** Ranks only timestamp and
   buffer; all attribution logic runs in the collector, where data from every
   rank is available.
2. **Batched, lossy-tolerant transport.** Missing 0.1% of timing records is
   fine; blocking a training step is not. Never do I/O on the hot path —
   the hook appends to a ring buffer and a background thread flushes.
3. **Offline-first data model.** The same record format feeds the live
   dashboard and a Parquet file for post-hoc analysis.

## 5. Data Model

Each collective invocation, on each rank, produces one record:

```json
{
  "job_id": "train-run-42",
  "rank": 147,
  "world_size": 512,
  "seq": 10382,
  "op": "all_reduce",
  "bucket_bytes": 104857600,
  "t_enqueue_ns": 1733440800123456789,
  "t_start_ns":   1733440800130000000,
  "t_end_ns":     1733440800455000000,
  "device": "cuda:3",
  "hostname": "node-18",
  "pg": "default"
}
```

Notes:

- `seq` is a per-process monotonically increasing counter. Combined with
  `(op, bucket_bytes)` it lets the collector line up the *same* collective
  across ranks. (Perfect cross-rank collective matching is harder than it
  looks — see Pitfalls.)
- `t_enqueue` (when the op was issued) vs `t_start` (when the kernel actually
  began) exposes queueing delay separately from execution time.
- `hostname` enables topology correlation (ranks 40–47 on the same node
  slowing down together = network, not compute).

## 6. Phase 1 — MVP: PyTorch Hook + Heatmap

Everything you need for the MVP is exposed by PyTorch. No NCCL internals.

### 6.1 The timing hook

The clean entry point is a wrapper around collective calls on the default
process group. Two viable approaches:

**Option A — monkey-patch `ProcessGroup` methods** (simplest, works
everywhere):

```python
import time
import torch.distributed as dist

class CollectiveTimer:
    def __init__(self, sink, rank, world_size):
        self.sink = sink          # ring buffer + background flusher
        self.rank = rank
        self.world_size = world_size
        self.seq = 0

    def wrap_all_reduce(self, original):
        def timed(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
            rec = {
                "rank": self.rank, "world_size": self.world_size,
                "seq": self.seq, "op": "all_reduce",
                "bucket_bytes": tensor.numel() * tensor.element_size(),
                "t_enqueue_ns": time.perf_counter_ns(),
            }
            self.seq += 1
            rec["t_start_ns"] = time.perf_counter_ns()
            work = original(tensor, op=op, group=group, async_op=async_op)
            # CPU-side return is async; true end time needs the CUDA event
            # approach below. For the MVP, wrapping work.wait() gets you
            # end-to-end latency as observed by this rank.
            if async_op:
                return _wrap_work(work, rec, self.sink)
            rec["t_end_ns"] = time.perf_counter_ns()
            self.sink.emit(rec)
            return work
        return timed
```

**Option B — CUDA events for GPU-true timing** (needed once you care about
kernel-level accuracy): record a CUDA event before and after the collective on
the current stream, and read the elapsed time on completion via a host
callback. This is what you graduate to when CPU timestamps turn out to measure
"when the CPU asked," not "when the GPU ran." For the MVP, Option A plus a
`torch.cuda.synchronize()` in a *debug-only* mode is fine — understand what
each measures before optimizing.

**Option C — `torch.distributed` flight recorder / Work-level introspection**:
newer PyTorch exposes per-Work timing and NCCL flight recorder dumps
(`TORCH_NCCL_DUMP_ON_TIMEOUT`, `torch._C._distributed_c10d`). Phase 2
territory.

### 6.2 Sink: ring buffer + batch flush

```python
class BufferedSink:
    def __init__(self, flush_fn, capacity=4096, flush_interval_s=2.0):
        self.buf = collections.deque(maxlen=capacity)
        self.flush_fn = flush_fn
        self.flush_interval_s = flush_interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def emit(self, rec):
        self.buf.append(rec)          # non-blocking, hot path safe

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(self.flush_interval_s)
            batch = list(self.buf)
            self.buf.clear()
            if batch:
                try:
                    self.flush_fn(batch)
                except Exception:
                    pass               # losing records > blocking training
```

### 6.3 Collector & the first useful analysis

The collector receives batches from all ranks, groups records by collective
instance (matching on `seq` windows + `op` + `bucket_bytes`), and computes the
single most valuable statistic:

**Arrival skew and exit skew per rank.**

For one collective across ranks, with `t_start[i]` and `t_end[i]`:

- `arrival_skew[i] = t_start[i] - min(t_start)` — how late rank *i* showed up.
  Persistently late arriver ⇒ that rank is **compute/CPU-slow** (data loader,
  host bottleneck), and every other rank burned time waiting *before* the
  collective even started.
- `exit_skew[i] = t_end[i] - min(t_end)` — how late rank *i* finished relative
  to the earliest finisher. Ranks that arrive on time but exit late indicate
  **network-slow** paths — their slices of the ring/tree are the bottleneck.

This distinction — *late to arrive* vs *late to leave* — is the core insight
of the whole tool, and it falls out of two timestamps. Phase 1 ships with just
this and it's already more attribution than most clusters have.

### 6.4 MVP dashboard

Keep it boring. A heatmap:

- X axis: time (collective sequence number)
- Y axis: rank
- Color: total collective duration (or exit skew)

A persistent straggler appears as a hot horizontal row. Jitter appears as
scattered hot cells. Topology-correlated slowness appears as hot *bands*
(ranks grouped by node/switch on the Y axis). One Streamlit or Grafana panel
is enough for the demo.

**MVP definition of done:** run a 2–4 GPU training job, artificially delay one
rank (see §9), and show the heatmap flagging it within seconds — plus a
printed summary: "rank 2 was last arriver in 78% of all-reduces over the last
10 minutes (median arrival skew 41 ms)."

## 7. Phase 2 — NCCL Flight Recorder Integration

PyTorch's NCCL backend has a **flight recorder**: an in-memory ring buffer of
collective events per process (entry/exit timestamps, sizes, and in recent
versions transport/channel metadata), originally built for debugging hangs and
timeouts.

Phase 2:

- Enable it (`TORCH_NCCL_TRACE_BUFFER_SIZE`, `TORCH_NCCL_DEBUG_INFO_TEMP_FILE`
  — exact knobs vary by version; check your PyTorch/NCCL release notes).
- Periodically dump and parse the traces in the sink's background thread.
- Merge flight-recorder events with Phase 1 records: the hook tells you *when*
  and *how slow*; the flight recorder tells you *which algorithm and channels*
  NCCL used (ring vs tree, NVLink vs IB, number of channels) — letting you
  say "this slowdown coincides with NCCL falling back to a tree over the
  network" rather than just "slow."

This is the phase where you learn NCCL's internals for real: channel
topology, P2P vs SHM vs NET transports, and why message size changes the
algorithm choice.

## 8. Phase 3 — NIC-Level Correlation

Software timing can tell you a rank is network-slow but not *why*. The last
layer correlates collective stalls with hardware counters:

- **RDMA/NIC counters**: retransmits, out-of-sequence packets, link flaps,
  per-port throughput (`perfquery` on InfiniBand, `ethtool -S` / devlink on
  RoCE/Ethernet). Poll these on each node; when a slow-collective alert fires
  on ranks attached to port X, check port X's counter deltas in the same
  window.
- **eBPF** (stretch): trace RDMA verbs or TCP flows for NCCL's transport,
  correlating packet-level events with collective windows. This is the
  "distributed tracing for a collective" endgame.

A concrete killer demo for this phase: a degraded transceiver throwing
retransmits is caught by the counter correlation even though every software
timestamp just says "rank 147 exits late."

## 9. Testing: Simulating Stragglers and Jitter

You can't build a straggler detector without stragglers. Cheap ways to make
them:

- **CPU delay injection**: `time.sleep(random.expovariate(...))` on one rank
  before collectives → classic straggler / jitter, arrives-late profile.
- **Network degradation**: `tc qdisc add dev eth0 root netem delay 20ms 5ms`
  on one node → arrives-on-time, exits-late profile.
- **Bandwidth cap**: `tc qdisc ... tbf rate 1gbit` → subtler, size-dependent
  slowdown.
- **No-GPU development**: PyTorch's **gloo** backend runs collectives on CPU,
  so you can develop and test the entire pipeline on a laptop with multiple
  processes before touching a GPU cluster. Strongly recommended for early
  iteration.
- **Real multi-node**: two cheap cloud GPU instances (or one node with
  multiple GPUs) is enough for every phase except large-scale topology
  correlation.

## 10. Pitfalls (the parts that will actually hurt)

1. **Clock skew across nodes.** `arrival_skew` compares timestamps from
   different machines. NTP gets you ~milliseconds; a slow all-reduce is
   measured in milliseconds. Either (a) estimate per-pair clock offsets by
   exchanging ping messages at startup and subtracting, or (b) compute skew
   *relative* to per-collective min and treat absolute offsets as noise —
   workable because stragglers show up as *consistent* ordering, which clock
   offset doesn't change much. Be honest in the docs about precision.
2. **Cross-rank collective matching.** With multiple process groups (FSDP
   creates many), bucketing, and async execution, "the same collective" is
   non-trivial to identify. Start with single-PG DDP; document the limitation.
3. **Async execution.** CUDA collectives are launched asynchronously. CPU
   timestamps measure issue time, not GPU execution time. Phase 1 accepts
   this; Phase 1.5 adds CUDA events. Never silently conflate the two.
4. **Overhead discipline.** The hot path is `append` to a deque — keep it
   that way. No logging calls, no string formatting, no locks on the emit
   path. Measure your own overhead with a before/after step-time benchmark
   and put the number in the README.
5. **Data volume.** 512 ranks × a collective every few ms × multiple PGs =
   firehose. Aggregate aggressively: the collector mostly needs per-window
   percentiles per rank, not every raw record forever.

## 11. Learning Path (what you actually get out of this)

| Phase | Skills you pick up |
|-------|--------------------|
| 1 | PyTorch distributed internals, CUDA async execution model, Python perf discipline, time-series viz |
| 1.5 | CUDA events & streams, GPU/CPU timeline alignment |
| 2 | NCCL architecture: rings, trees, channels, transports; parsing opaque binary/debug formats |
| 3 | RDMA & InfiniBand/RoCE fundamentals, NIC counters, eBPF tracing, network forensics |
| Throughout | Distributed systems observability: clock sync, attribution, aggregation under scale |

## 12. Existing Work to Study (build anyway — that's the point)

- **nccl-tests** — baseline bandwidth benchmarks; useful as a sanity harness.
- **PyTorch NCCL flight recorder** — your Phase 2 data source; read its dump
  format early.
- **Kineto / PyTorch profiler** — how a mature per-node tracer structures
  GPU timeline data.
- **MegaScale (ByteDance paper, 2024)** — production report on diagnosing
  stragglers at 10k-GPU scale; shows what the state of the art actually
  monitors, and how much of it is ad hoc.
- **Grafana/Prometheus ecosystem** — don't reinvent storage and dashboards;
  emit metrics in a format they ingest.

The point of the project isn't novelty — it's that the full loop (hook →
transport → attribution → visualization → hardware correlation) touches real
problems that most people only ever see one slice of. Building it end-to-end,
even where pieces exist, is where the learning is.

## 13. Suggested Milestones

1. **Week 1**: gloo-backend prototype on one machine, 4 processes; ring
   buffer sink; Parquet dump; offline heatmap in a notebook.
2. **Week 2**: real GPU run (single node, multi-GPU); injected straggler;
   live dashboard; "last arriver %" alert.
3. **Week 3**: CUDA-event timing; overhead benchmark; two-node run with
   `tc netem` network straggler; arrival-vs-exit classification working.
4. **Week 4**: flight recorder parsing; algorithm/transport metadata in the
   UI.
5. **Beyond**: NIC counter correlation, eBPF, write-up + demo.

A good demo video (planted straggler caught in real time on the heatmap) is
the artifact that makes this legible to other people — plan for it from the
start.
