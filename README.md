# biomni-profiling

Workload characterization of the CPU/GPU scheduling bubble in agentic AI,
using Biomni (snap-stanford) as the benchmark workload and Biomni-R0 (32B)
served locally on du04.

## What is being measured

A Biomni agent program is a LangGraph loop of two phases:

- **generate** — the LLM writes a plan / code. Reasoning runs on Biomni-R0
  via SGLang across the 4 A100s, so this phase is **GPU-busy**.
- **execute** — the generated Python/R/Bash runs in a persistent REPL. For
  CPU-bound bio tools this phase is **CPU-busy and GPU-idle** — the GPU bubble.

The loop is `generate -> execute -> generate -> execute -> ...` until a final
answer. This repo's tracer records the boundaries of those two phases plus
hardware utilization, on one monotonic clock, so the bubble can be seen.

## Layout

```
biomni-profiling/
├── README.md                 this file
├── serve_biomni_r0.sh        launches the SGLang server (long-lived)
├── trace_run.py              runs ONE agent program with hooks + hardware sampling
└── traces/                   outputs, one timestamped subfolder per run
    └── <UTC-timestamp>_run/
        ├── events.jsonl      generate/execute phase boundaries
        ├── hardware.csv      per-GPU + per-CPU samples (100ms default)
        ├── summary.json      turns, per-phase durations, layer-1 bubble estimate
        ├── meta.json         run config + wall-clock anchor
        └── agent_output.txt  the agent's final answer (sanity check)
```

Biomni itself is a pip package inside the `biomni-sglang` conda env — it is
**not** edited. The tracer wraps its functions from the outside by import path.
The 32B weights live in the HF cache at
`/DATA/mansari26/huggingface_cache/hub/`.

## Running a trace (two terminals on du04)

### Terminal 1 — start the model server (leave it running)

```bash
conda activate biomni-sglang
cd biomni-profiling
bash serve_biomni_r0.sh
```

Wait for SGLang to report the server is ready (endpoint at
`http://localhost:30000/v1`). This holds all 4 GPUs.

### Terminal 2 — run trace #1

```bash
conda activate biomni-sglang
cd biomni-profiling
python trace_run.py
```

This runs the default simple task (ADMET property prediction — uses RDKit
in-process, no data lake) once, then writes the trace into `traces/`.

The server in Terminal 1 stays up; re-run `trace_run.py` for more traces.

## Dependencies for the tracer

In the `biomni-sglang` env:

```bash
pip install pynvml psutil
```

`pynvml` (per-GPU utilization) and `psutil` (per-CPU). If `pynvml` is missing
the tracer still records CPU and warns; GPU columns will be blank.

## What trace #1 is for

Trace #1 validates the machinery, not the science. Confirm:

1. the server serves R0 under `--tp 4` and the agent connects;
2. `events.jsonl` shows alternating generate/execute phases (the loop shape);
3. both hooks fire — if the generate hook can't find the agent's LLM object,
   the tracer says so and we fix the attribute name from the real `A1`.

Once the structure looks right, trace #2 drives a CPU-bound tool (e.g. a
microbiome / scRNA-seq task) so the CPU-busy phase and GPU bubble appear in
`hardware.csv`, and the full Tier-1 profiler is built around what we observe.
