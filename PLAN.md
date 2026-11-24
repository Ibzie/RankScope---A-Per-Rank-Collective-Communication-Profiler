# RankScope — Implementation Plan

**A per-rank collective communication profiler.**
Live document. The original (uncorrected) design document is archived at
[`docs/nccl-profiler-design-doc.md`](docs/nccl-profiler-design-doc.md); this file
supersedes it.

Last updated: 2026-11-24 · Status: Step 0

---

## 1. What this is

Distributed training is a sequence of collectives (all-reduce, all-gather,
reduce-scatter). A collective is a barrier: no rank proceeds until every rank
contributes. So the cluster runs at the speed of its slowest link at every step,
and two failure modes cost real money:

- **Stragglers** — one rank is persistently slow (degraded NIC, thermal
  throttling, bad cable, bad route). Everything waits on it, forever, and no
  standard dashboard points at it.
- **Jitter** — no rank is consistently slow, but some rank is randomly slow
  (OS noise, checkpoint writes, dataloader hiccups). Invisible per event,
  enormous in aggregate.

GPU utilisation dashboards actively mislead here: a GPU blocked in an
all-reduce still reports as "busy".

The tool answers three questions:

1. **Which** rank slowed down this collective?
2. **Why class** of problem — arrived late (compute/CPU-slow) or arrived on time
   and exited late (network-slow)?
3. **Persistent or transient**, and is it topologically clustered (same node,
   same switch)?

The second question falls out of two timestamps and is already more attribution
than most clusters have.

## 2. Hardware and the constraint that shapes everything

| | |
|---|---|
| Desktop GPU | 1× RTX 4060, 8 GB, `sm_89`, ~272 GB/s |
| Laptop GPU | 1× RTX 4060, CachyOS (same as desktop) |
| Desktop board | MSI B550M-A PRO, 12 cores, 31 GB RAM |
| Desktop NIC | RTL8111 (`r8169`), gigabit-capable; negotiated 100 Mb/s only because the router advertises 10/100 |
| Inter-node link | Direct Ethernet cable, USB RJ45 adapter + laptop port, 1 GbE |
| Inter-node data path | **soft-RoCE (`rdma_rxe`)** — NCCL has no TCP data transport |
| Kernel | 7.3.5-2-cachyos (`rdma_rxe.ko` present, not loaded) |
| Python | 3.12 via `uv` (system 3.14 is too new for torch) |
| torch | 2.14.1+cu130, pinned in `uv.lock`, identical on both nodes |

**The hard constraint:** NCCL needs one *distinct device* per rank. Two ranks
on `cuda:0` abort with `Duplicate GPU detected`. This machine has one GPU, so
any real collective needs the second machine. That is why the plan is
*CUDA-first but single-rank-first*: everything about timing can be learned and
validated on one GPU, and the multi-rank layer is added when the cable exists.

**The corollary that matters for a profiler:** the transport's *mean* latency is
irrelevant; its *variance* is everything. A 10–40 ms injected straggler against
a wired p99 of 0.4 ms is a 20–100× signal-to-noise ratio. Against WiFi's
20–100 ms p99 the noise *is* the signal, and the tool would produce convincing
false stragglers. Slow and boring beats fast and noisy. Never profile a
collective over WiFi.

## 3. Corrections to the original design document

These are the substantive errors in the archived doc. They are listed first
because they determine the architecture.

1. **`perf_counter_ns()` around `dist.all_reduce()` measures nothing.** With CUDA
   tensors the call returns at *enqueue*. You are timing Python and driver
   overhead, typically microseconds, for a collective that took milliseconds.
2. **`work.wait()` conflates.** It blocks until everything queued on the stream
   drains, so it silently attributes other kernels' runtime to your collective.
   It measures "when the host noticed", not "when the collective ran".
3. **A Python hook cannot see DDP.** DDP's reducer calls
   `ProcessGroupNCCL::allreduce` in C++, so monkey-patching the Python method
   intercepts nothing. For real training scripts the data source must be Kineto
   or the NCCL flight recorder. The hand-written hook is for collectives *we*
   call, which is exactly what the labs need.
4. **CUDA event timestamps are per-device clocks.** They are not comparable
   across GPUs, and `cudaEventElapsedTime` is only valid within one device.
   Cross-rank skew therefore has to come from host `CLOCK_MONOTONIC`, which *is*
   machine-wide and therefore directly comparable across processes on one node.
   The reconciliation is: event durations per rank, host clocks for cross-rank
   skew, and back-projection
   `t_start ≈ t_end_observed − gpu_duration`.
5. **NCCL has no TCP data path.** `NCCL_SOCKET_IFNAME` selects the bootstrap
   interface only. Inter-node bulk data requires a `NET` transport
   (ibverbs: InfiniBand, RoCE, EFA). Without an RDMA device, a 2-node job prints
   `NET/IB : No device found` and then hangs. Soft-RoCE `rxe` provides an
   ibverbs device over ordinary UDP.

## 4. Principles

1. **Collect at the source, judge at the centre.** Ranks timestamp and buffer;
   all attribution logic runs where every rank's data is visible.
2. **Nothing on the hot path but an append.** No formatting, no logging, no
   locks, no I/O. Flush from a background thread.
3. **Measure the instrument.** Every layer of the profiler gets benchmarked
   against a no-profiler baseline, and the number goes in the README.
4. **Never conflate two clocks.** Every timestamp in a record is labelled with
   what it actually measures and which clock domain it lives in.
5. **One experiment per question.** If a measurement can fail for two reasons,
   split it.

## 5. Steps

| Step | Content | Needs hardware | Status |
|---|---|---|---|
| 0 | Env, GPU ground truth, two-ranks-one-GPU spike | desktop only | **in progress** |
| 0.5 | Two-node bring-up ladder | cable + adapter | blocked |
| 1 | What is a timestamp (stream truth lab) | desktop only | todo |
| 2 | Ring all-reduce, hand-written in CUDA | desktop only | todo |
| 3 | Profiler core + synthetic multi-rank replay | desktop only | todo |
| 4 | Two-node NCCL, straggler hunt, clock offsets | both nodes | todo |

### Step 0 — Ground truth

Establish the facts the rest of the project is built on, and prove the
constraint rather than assuming it.

*Deliverable:* `labs/00_hello_gpu.py`, which reports

- torch / CUDA / **bundled NCCL** versions (the NCCL version decides every
  algorithm choice later, so it belongs in the record)
- device name, compute capability, SM count, L2 size, memory
- **measured** device-to-device copy bandwidth — our own roofline, not a spec
  sheet
- a `world_size=1` all-reduce baseline across sizes, event-timed: the
  degenerate case everything else is compared against
- the two-ranks-one-GPU spike, in a subprocess, expected to die with
  `Duplicate GPU detected`

*Done when:* the lab runs clean and the spike fails for the *expected* reason.

### Step 0.5 — Two-node bring-up

Five rungs, each isolating exactly one failure mode. Do not skip a rung to save
time; the whole point is that a failure tells you *where* you are.

| Rung | Command | Proves |
|---|---|---|
| 0 | `lsusb -t`, `ethtool -i`, `ethtool \| grep Speed`, `iperf3` | adapter and link are sane |
| 0b | `ping -c 20000 -i 0.005` | **noise floor** — min/avg/max/mdev, count > 1 ms |
| 1 | `ping -M do -s 8972` | IP + jumbo frames, no RDMA involved |
| 2 | `rdma link show`, `ibv_devinfo -v` | rxe attached, GID table readable |
| 3 | `qperf 10.10.10.2 rc_bw` | **libibverbs** path works |
| 4 | 2-rank NCCL `all_reduce`, 1 MB | NCCL's `NET` transport works over rxe |

Rung 3 is the one people skip and regret: it exercises RoCE without PyTorch or
NCCL in the picture, so a rung-4 failure is unambiguously an NCCL config
problem.

*Done when:* rung 4 completes and the rung 0b distribution is recorded in
`docs/notes-link.md`. Those numbers become the detector's noise floor.

### Step 1 — What is a timestamp

The intellectual core of the whole tool, done before any profiler exists.

Five experiments in `labs/01_stream_truth.py`:

1. Time a kernel from the host → wrong by the launch latency.
2. Time it with CUDA events → correct.
3. Enqueue 8 kernels back to back → host returns in µs, GPU takes ms. *This is
   the firehose.*
4. Preload a 20 ms dummy kernel, then time the next kernel from the host → it
   looks instant. Events show the truth. **This is exactly what a straggler
   looks like.**
5. Poll `event.query()` instead of `synchronize()` → measure the polling
   latency, and learn why syncing on the hot path is unacceptable.

*Deliverable:* the lab plus `docs/notes-timestamps.md`, which becomes the
specification for the record format.

*Done when:* each experiment's number is explained in the note, and we can state
in one sentence what `t_enqueue`, `t_start` and `t_end` each measure.

### Step 2 — Ring all-reduce, hand-written

You cannot profile an algorithm you cannot implement.

- `reduce_scatter_hop` and `all_gather_hop` kernels in one `.cu` file, compiled
  via `torch.utils.cpp_extension.load_inline`. Vectorised `float4` loads,
  grid-stride loops, ~80 lines total.
- A Python driver that loops the `2(N−1)` steps, with `N` **logical** ranks
  mapped onto slices of one buffer, and "send to rank r+1" implemented as a
  kernel hop. It prints the schedule as ASCII *before* it runs.
- Validation against `torch.sum` — the algorithm is proven correct, not merely
  plausible.
- Byte accounting: ring moves `2S(N−1)/N` per rank; gather-to-root-and-broadcast
  moves `2S(N−1)` through the root. That ratio is why rings exist.
- The punchline: on one GPU the ring is **slower** than a single fused kernel,
  because you pay `2(N−1)` kernel launches to teach a lesson. That is precisely
  why NCCL ships persistent cooperative kernels.
- *Stretch:* the same schedule as one cooperative kernel, blocks = ranks,
  `grid.sync()` = the network.

*Done when:* the ring output matches `torch.sum` and the exploit/inefficiency
crossover is measured, not asserted.

### Step 3 — Profiler core

- Event pool (preallocated, reused) → `BufferedSink` (deque + background
  flusher) → `CollectiveRecord` → parquet.
- Back-projected host start, per the reconciliation in §3.4.
- Arrival skew and exit skew; "last arriver in X% of collectives" summary.
- Detector thresholds derived from the measured noise floor, not guessed.
- Developed against **replayed synthetic multi-rank traces**, because the
  analysis layer needs many ranks while the recorder needs real GPUs.

*Done when:* synthetic traces containing a known straggler are flagged
correctly, and a synthetic jitter trace is *not* flagged as a straggler.

### Step 4 — Two-node straggler hunt

- Hook real collectives; inject stragglers **GPU-side** by launching a long
  kernel on one rank's stream before the collective. That produces real arrival
  skew; `time.sleep` produces a different, weaker phenomenon.
- NTP-style clock-offset estimation between the two hosts, validated against a
  known injected delay.
- Live heatmap: rank × collective index, coloured by duration or exit skew.

*Done when:* a planted straggler appears as a hot row within seconds, and the
summary names it.

## 6. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-11-24 | No gloo/CPU phase | CUDA async semantics are the subject; gloo would teach the wrong thing |
| 2026-11-24 | CUDA events for durations, host clocks for cross-rank | Event clocks are per-device; `CLOCK_MONOTONIC` is machine-wide |
| 2026-11-24 | soft-RoCE over direct Ethernet | NCCL has no TCP data path; WiFi variance exceeds the signal |
| 2026-11-24 | USB adapter, not PCIe | Portable between the two machines; also lets us choose which end of the link is the fast one |
| 2026-11-24 | torch 2.14.1 pinned via `uv.lock` | Mismatched NCCL versions across nodes hang silently |
| 2026-11-24 | Buffers 4–32 KB on 100 Mb/s, 32–512 KB on 1 GbE | Keeps us in the latency-bound regime where skew dominates |

## 7. Glossary

- **Collective** — an operation all ranks participate in. `all_reduce` sums
  every rank's buffer and gives the result to everyone.
- **Rank / world size** — a participant's index and the number of participants.
- **Ring all-reduce** — split the buffer into `N` chunks; `N−1` steps of
  reduce-scatter (each rank accumulates one chunk as it circulates), then `N−1`
  steps of all-gather (each reduced chunk circulates back). Cost per rank is
  `2S(N−1)/N`, i.e. roughly independent of `N` — no hot spot.
- **Tree** — a reduction tree, then a broadcast. Fewer sequential hops than a
  ring for small messages; NCCL picks between the two by message size.
- **Bucket** — DDP coalesces many small gradients into one large collective to
  amortise latency. Bucket boundaries determine what a "collective" even is,
  which is why cross-rank matching is hard.
- **Transport** — NCCL's data movers: `P2P` (NVLink/PCIe), `SHM` (same node),
  `NET` (ibverbs: IB/RoCE/EFA).
- **RoCE / rxe** — RDMA over Converged Ethernet; `rxe` is the in-kernel
  *software* implementation, which gives you a RoCE device on any netdev.
- **GID** — the address an RDMA device uses. Wrong GID index = silent hang.
- **Flight recorder** — NCCL's in-memory ring buffer of recent collective
  events per process, with algorithm and channel metadata. Phase 2 data source.
- **Arrival skew vs exit skew** — late to start vs late to finish. The first is
  compute-slow, the second is network-slow.

## 8. Risks and open questions

- **Adapter quality.** Cheap USB NICs sometimes log `rx/tx timeout` and reset.
  A link flap looks exactly like jitter in our data. Mitigation: rung 0 and 0b,
  and treat resets as a known artifact in the notes rather than a finding.
- **Autosuspend** is currently `auto` on this box and must be forced off for the
  adapter, or it will inject multi-millisecond stalls.
- **Clock skew across the two hosts** is real, not theoretical. NTP on a LAN is
  ~1–5 ms, which is the same order as the signal. Needs the ping-based offset
  estimator in Step 4 before any cross-node skew number is trusted.
- **Laptop port speed unknown** — decides topology A (adapter on desktop) vs
  B (adapter on laptop). Does not block anything.
- **`rxe` is software RDMA**, so CPU cost is real and single-queue. Fine at
  1 GbE; it would not be fine at 100 GbE.
