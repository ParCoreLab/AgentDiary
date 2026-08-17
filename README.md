# AgentDiary

**A four-layer profiler that measures the CPU/GPU "bubble" in agentic AI workloads** —
where an LLM agent's wall-clock time actually goes, turn by turn, across GPU
inference and CPU tool execution — and a set of experiments that turn that
measurement into a causal, quantitative story about what actually bottlenecks
agent serving at scale.

The workload used throughout is [Biomni](https://github.com/snap-stanford/Biomni)
(Stanford SNAP's biomedical research agent), driving a real 32B-parameter model
(Biomni-R0) served locally via [SGLang](https://github.com/sgl-project/sglang)
— not a toy benchmark, not simulated tool calls.

## The problem

An agent isn't one LLM call — it's a loop: the model reasons and picks an
action (**generate**, GPU-busy), then a tool actually runs (**execute**,
typically CPU-busy). Because a single program runs one phase at a time, each
phase idles the *other* resource: the GPU sits idle during tool execution,
the CPU sits idle during inference. That idle window is the **bubble**.

Existing agent-serving schedulers (Autellix, Continuum, and similar) optimize
GPU occupancy and treat everything off-GPU — a tool call, a network wait, a
database read — as one opaque "the program is away" interval. That throws
away exactly the signal this profiler is built to capture: **what kind of
work is happening off-GPU, how much of it, and how does it respond to
contention.**

## Key findings

**CPU contention causally bounds the tool-execution phase — not the GPU.**
A controlled sweep (fixed workload, only available CPU cores varied via
`taskset`, GPU serving pinned to its own cores so it's never starved) shows
execute time scaling almost exactly as 1/cores once available cores drop
below the workload's actual demand — while GPU serving metrics (TTFT, TPOT,
KV-cache occupancy) stay flat throughout, ruling out a GPU-side explanation:

| Cores available | Execute time vs. baseline |
|---|---|
| 120 / 72 / 40 (at or above demand) | flat, 1.0–1.15× |
| 24 | 2.0× |
| 12 | up to 5.3× (one task: exactly 2.00× when cores were halved 24→12) |

**GPU-only scheduling can be blind to a saturated host.** At 100 concurrent
agents against one shared server, CPU hit ~98% (157/160 logical cores) while
GPU serving still had headroom (KV-cache 87%, request queue ≈0) — a scheduler
watching only GPU-side signals would read "room for more tenants" exactly
when the host is already the bottleneck. Fixed-work tool phases measurably
lengthened 28–42% purely from this contention. Pushed to 200 agents, CPU
saturated completely and the failure mode shifted to host memory — the
kernel OOM-killer terminated the most memory-hungry agents, while GPU still
hadn't hit its own wall.

**Decode is memory-bandwidth-bound, not compute-bound.** During generation,
GPU SM activity runs ~66% while SM *occupancy* stays ~6% — the SMs are
active but stalled waiting on HBM bandwidth, not FLOPs. TPOT stays ~17–18ms/
token, TTFT ~1–1.3s, largely invariant to context length.

**Per-agent CPU load is light in cores but dominant in latency.** A single
agent's tool phase typically saturates only 2–9 of 256 available cores (one
hot core, everything else idle) — yet that phase is up to 87% of total
wall-clock time for CPU-heavy tasks. This reconciles a real tension in how
"CPU-bound" gets used: light in *utilization*, dominant in *time-on-the-
critical-path* — consistent with industry reporting that tool processing can
account for up to 90.6% of agentic request latency.

**The bubble is workload-invariant across hardware.** The same task run on
two different GPU generations (A100-40GB vs. H100-64GB) produced nearly
identical bubble fractions (87.3% vs. 86.9%) — the bubble is a property of
the *workload's* phase structure, not a hardware-specific artifact.

## How it works

Every trace joins four measurement layers on one shared clock
(`time.perf_counter()`), so any row in any output file can be correlated
against any other by timestamp:

| Layer | Captures | How |
|---|---|---|
| **L1** — agent loop | generate/execute phase start & end | wraps the agent's LLM-call and tool-exec functions from the outside, by import path — the agent framework's own code is never modified |
| **L2** — HTTP | per-call latency, prompt/completion tokens | extracted from the same wrapped call, no extra round-trip |
| **L3** — serving engine | TTFT, TPOT, queue depth, KV-cache occupancy | scrapes the inference server's Prometheus `/metrics` endpoint |
| **L4** — hardware | per-GPU SM activity/occupancy/DRAM/power, per-core CPU | DCGM (falls back to NVML if unavailable) + `psutil` |

For multi-agent runs, L3/L4 use **one shared sampler per session**, not one
per agent — an earlier per-agent-scraper design was found to flood the
serving endpoint's metrics port and corrupt the very latency numbers it was
trying to measure. Worth knowing if you extend this: sampling infrastructure
that reads the system can itself become a confound at concurrency.

## Repo layout

```
profiling/          the tracer + analysis:
  trace_run.py         runs one agent, produces the 4-layer trace
  analyze_trace.py      per-trace metrics + figures
  aggregate_traces.py    cross-trace rollup
  plot_concurrent.py, plot_sustained.py    multi-agent figures

scripts/            server launch + multi-agent orchestration
  serve_biomni_r0.sh    starts the SGLang server
  run_concurrent.py, run_sustained.py    N-agent drivers

tasks/              task configs (prompt + resource-profile label per task)
  _deprecated_fake/     early synthetic tasks superseded by real Biomni-tool tasks

session_configs/    multi-agent experiment configs (which tasks, how many, stagger timing)
```

Running any of the above writes into `results/` (single-agent) or
`results_multi/` (multi-agent) — created locally, not checked into the repo
(trace data is large and fully regenerable from the code here).

## Running it yourself

This is written to run on any machine with enough combined GPU memory to
serve a ~32B-parameter model — it isn't tied to any specific cluster. It was
developed and validated on two: a 4×A100-40GB node and a 4×H100-64GB node
(see the cross-hardware finding above); the setup below applies to either,
or to comparable hardware of your own.

**Requirements**
- 4 GPUs with enough combined memory for a 32B model in bf16 (~63GB total,
  so ≥16GB/GPU on 4 cards) — fewer/larger GPUs work too, adjust `--tp` in
  the launch script to match your GPU count
- Python 3.11, [SGLang](https://github.com/sgl-project/sglang) installed
- Optional: [DCGM](https://developer.nvidia.com/dcgm) for the richer
  hardware metrics (SM occupancy, DRAM activity); the tracer falls back to
  NVML automatically if DCGM isn't set up

**Setup**
```bash
conda create -n biomni-sglang python=3.11 -y
conda activate biomni-sglang
pip install sglang biomni psutil pynvml
```

**Before your first run**, `scripts/serve_biomni_r0.sh` hardcodes two paths
that are specific to the machine it was developed on — open it and adjust:
- `CUDA_HOME` — point this at your own CUDA toolkit install
- `HF_HOME` — point this at wherever you want model weights cached

**Launch the server** (long-lived, one terminal):
```bash
conda activate biomni-sglang
bash scripts/serve_biomni_r0.sh
# ready when it prints: "The server is fired up and ready to roll!"
```

**Run a trace** (separate terminal, server stays up):
```bash
conda activate biomni-sglang
python profiling/trace_run.py --task-config tasks/gillespie_microbiome.json
python profiling/analyze_trace.py results/gillespie_microbiome/<timestamp>/
```

Swap in any config under `tasks/` for a different workload, or write your
own — a task config is just a prompt plus a resource-profile label.

## Status

The measurement layer and the CPU-throttle causal result above are validated
across single- and multi-agent runs on two hardware generations. What's
*not* in this repo yet: a scheduler that acts on this signal — that's the
next phase of the work, not a finished claim.
