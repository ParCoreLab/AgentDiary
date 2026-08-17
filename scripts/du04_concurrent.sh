#!/bin/bash
# =============================================================================
# du04 concurrent launcher (NO SLURM) — the FIXED, shared-sampler multi-agent runner for du04.
# du04 runs directly (not via a scheduler), DCGM is a system service, and the SGLang server is
# long-lived in another terminal, so this just checks the server is up, generates the session
# config, and runs the shared-sampler run_concurrent.py.
#
# 1) Terminal A (once):   conda activate biomni-sglang && bash scripts/serve_biomni_r0.sh
# 2) Terminal B:          conda activate biomni-sglang && bash scripts/du04_concurrent.sh <N> <task> [stagger_s]
#
# NOTE: du04 is A100-40GB — the 32B model leaves little KV headroom, so concurrency is limited to
# ~4–8 agents (a real hardware cap, independent of the profiler bug we fixed). Use MN5 for high-N.
# =============================================================================
set -uo pipefail
N="${1:?usage: bash scripts/du04_concurrent.sh <N_agents> <task_id> [stagger_s]}"
TASK="${2:?usage: bash scripts/du04_concurrent.sh <N_agents> <task_id> [stagger_s]}"
STAGGER="${3:-0}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
PORT=30000
export MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-1000}"
export MAX_LLM_CALLS="${MAX_LLM_CALLS:-20}"
# du04 DCGM is a system service; bindings at the default path (also trace_run's default).
export DCGM_BINDINGS_PATH="${DCGM_BINDINGS_PATH:-/usr/share/datacenter-gpu-manager-4/bindings/python3}"

echo "=== du04 concurrent: N=$N task=$TASK stagger=${STAGGER}s cap(calls=$MAX_LLM_CALLS,wall=${MAX_WALL_SECONDS}s) ==="
[ -f "$REPO/tasks/${TASK}.json" ] || [ -f "$REPO/tasks/_deprecated_fake/${TASK}.json" ] || { echo "ERROR: task ${TASK}.json not found in tasks/ or tasks/_deprecated_fake/"; exit 1; }
if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" != "200" ]; then
  echo "ERROR: SGLang server not responding at http://127.0.0.1:$PORT — start it first:"
  echo "       conda activate biomni-sglang && bash scripts/serve_biomni_r0.sh"; exit 1
fi
[ "$N" -gt 8 ] && echo "WARNING: N=$N on A100-40GB may exhaust KV cache; du04 realistically holds ~4-8 agents."

# resolve task path (tasks/ or the archived CPU-heavy set)
TPATH="tasks/${TASK}.json"; [ -f "$REPO/tasks/${TASK}.json" ] || TPATH="tasks/_deprecated_fake/${TASK}.json"

CFG=$REPO/session_configs/_du04_${N}x_${TASK}$([ "$STAGGER" != "0" ] && echo "_stag${STAGGER}").json
python - "$CFG" "$N" "$TPATH" "$TASK" "$STAGGER" <<'PY'
import json, sys, random
cfg_path, n, tpath, tid, stagger = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5])
random.seed(0); offs, acc = [], 0.0
for i in range(n):
    offs.append(round(acc, 2)); acc += random.expovariate(1.0/stagger) if stagger > 0 else 0.0
mode = f"Poisson arrivals mean {stagger:g}s" if stagger > 0 else "simultaneous (offset=0)"
agents = [{"agent_id": f"agent_{i:02d}_{tid}", "task_config": tpath,
           "arrival_offset_s": offs[i], "numactl_node": None} for i in range(n)]
json.dump({"session_name": f"du04_{n}x_{tid}" + (f"_stag{int(stagger)}" if stagger > 0 else ""),
           "description": f"du04 (A100-40GB): {n} '{tid}' agents, {mode}, shared-sampler (FIXED profiler).",
           "agents": agents}, open(cfg_path, "w"), indent=2)
print(f"[cfg] {cfg_path}: {n} agents, {mode}")
PY

cd "$REPO"
python scripts/run_concurrent.py "$CFG" --base-url "http://127.0.0.1:$PORT/v1"
echo "=== done — results in results_multi/ (shared-sampler; valid multi-agent numbers) ==="
