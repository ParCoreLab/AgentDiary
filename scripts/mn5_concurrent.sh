#!/bin/bash
# =============================================================================
# MN5 CONCURRENT launcher — run N agents of ONE task SIMULTANEOUSLY against one
# shared SGLang server. Simulates N users hitting the server at the same time.
#
# Server + DCGM start once; run_concurrent.py launches all N agents at offset=0,
# waits for all (each auto-analyzes), and writes a single-vs-multi comparison.
#
# Submit from an ACC login node:
#     sbatch mn5_concurrent.sh <N_agents> <task_id>
#   e.g.  sbatch mn5_concurrent.sh 10 gillespie_microbiome
#   faster queue (single session ~30-40 min):
#         sbatch --qos=acc_debug --time=02:00:00 mn5_concurrent.sh 10 gillespie_microbiome
# Watch:  tail -f concurrent_<JOBID>.log
# Result: results_multi/session_<ts>/  (per-agent traces + session_comparison.txt)
# =============================================================================
#SBATCH --job-name=biomni-concurrent
#SBATCH --account=etur02
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/projects/etur02/koc858886/biomni/concurrent_%j.log
#SBATCH --error=/gpfs/projects/etur02/koc858886/biomni/concurrent_%j.log

set -uo pipefail
N_AGENTS="${1:?usage: sbatch mn5_concurrent.sh <N_agents> <task_id> [stagger_s]}"
TASK="${2:?usage: sbatch mn5_concurrent.sh <N_agents> <task_id> [stagger_s]}"
STAGGER="${3:-0}"   # mean seconds between agent arrivals (Poisson); 0 = all launched at once

# RUNAWAY CAP (evidence-based, 2026-08-07). A legit gillespie's agent.go is ~200-500s even heavily
# loaded (50x clean median generate 335s, max well under 1000s), whereas a runaway generates ~100K
# tokens for ~1700s until it overflows the 131K context (20x/50x outliers ran 1700-2500s and owned
# the session). MAX_WALL_SECONDS is time-since-first-LLM-call, so 1000s kills a runaway while sparing
# slow-but-legit runs. Preferred over a call-count cap because 25+ calls can be verbose-legit OR runaway.
export MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-1000}"

BASE=/gpfs/projects/etur02/koc858886/biomni
REPO=$BASE/biomni-profiling
VENV=$BASE/venv
PORT=30000
SERVER_LOG=$BASE/server_${SLURM_JOB_ID}.log

echo "=== CONCURRENT job $SLURM_JOB_ID on $(hostname) | $(date) ==="
echo "=== N_agents=$N_AGENTS  task=$TASK  stagger=${STAGGER}s  MAX_WALL_SECONDS=${MAX_WALL_SECONDS}s ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
[ -f "$REPO/tasks/${TASK}.json" ] || { echo "ERROR: tasks/${TASK}.json not found"; exit 1; }

module load anaconda/2024.02
module load cuda/12.6
source "$VENV/bin/activate"
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export DCGM_ROOT=$BASE/dcgm
export LD_LIBRARY_PATH="$DCGM_ROOT/usr/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$DCGM_ROOT/usr/bin:$PATH"
export DCGM_BINDINGS_PATH="$DCGM_ROOT/usr/share/datacenter-gpu-manager-4/bindings/python3"
MODEL="RyanLi0802/Biomni-R0-Preview"

# ---- generate the session config: N copies of TASK with Poisson (or simultaneous) arrivals ----
CFG=$REPO/session_configs/_generated_${N_AGENTS}x_${TASK}$([ "$STAGGER" != "0" ] && echo "_stag${STAGGER}").json
python - "$CFG" "$N_AGENTS" "$TASK" "$STAGGER" <<'PY'
import json, sys, random
cfg_path, n, task, stagger = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
random.seed(0)
# arrival offsets: Poisson process (exponential inter-arrival, mean=stagger s). stagger=0 -> all at once.
offs, acc = [], 0.0
for i in range(n):
    offs.append(round(acc, 2))
    acc += random.expovariate(1.0 / stagger) if stagger > 0 else 0.0
mode = (f"Poisson arrivals, mean inter-arrival {stagger:g}s (last agent ~{offs[-1]:.0f}s)"
        if stagger > 0 else "all launched simultaneously (offset=0)")
agents = [{"agent_id": f"agent_{i:02d}_{task}",
           "task_config": f"tasks/{task}.json",
           "arrival_offset_s": offs[i],
           "numactl_node": None} for i in range(n)]
json.dump({"session_name": f"concurrent_{n}x_{task}" + (f"_stag{int(stagger)}" if stagger > 0 else ""),
           "description": (f"{n} '{task}' agents, {mode}, on one shared SGLang server. Staggered arrival "
                           f"removes the all-at-once startup pileup and exposes steady-state serving + "
                           f"orchestration-level CPU over time. Per-agent execute/generate/bubble/TTFT/TPOT + fill."),
           "agents": agents}, open(cfg_path, "w"), indent=2)
print(f"[cfg] wrote {cfg_path}: {n} agents, {mode}")
PY

# ---- start server (once) ----
echo "=== [setup] starting SGLang server (loads ~122GB, several min) ==="
# SGLANG_EXTRA_FLAGS lets a diagnosis run add server flags without editing this file.
# Method E (ground-truth per-request timing): SGLANG_EXTRA_FLAGS="--enable-request-time-stats-logging"
python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --host 127.0.0.1 \
    --tp 4 --dtype bfloat16 --mem-fraction-static 0.85 --trust-remote-code --enable-metrics \
    ${SGLANG_EXTRA_FLAGS:-} \
    --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":1.0,"original_max_position_embeddings":32768}, "max_position_embeddings": 131072}' \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
READY=0
for i in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server exited early:"; tail -n 30 "$SERVER_LOG"; exit 1; fi
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ]; then
    READY=1; echo "=== server ready after ~$((i*10))s ==="; break
  fi
  sleep 10
done
[ "$READY" -eq 1 ] || { echo "server never became ready"; kill "$SERVER_PID" 2>/dev/null; exit 1; }

# ---- DCGM hostengine (foreground -n, as a bg job; SLURM cgroup kills a daemonized one) ----
echo "=== [setup] starting DCGM hostengine ==="
nv-hostengine -n > "$BASE/hostengine_${SLURM_JOB_ID}.log" 2>&1 &
HE_PID=$!
sleep 5
if kill -0 "$HE_PID" 2>/dev/null; then echo "  DCGM hostengine up (pid $HE_PID)"; else echo "  DCGM failed -> Layer-4 uses NVML"; fi

# ---- launch all N agents SIMULTANEOUSLY ----
cd "$REPO"
# HW_INTERVAL controls the per-agent L3/L4 sampling period (default 50ms). For the 3c control
# run, set HW_INTERVAL=5.0 to throttle the 10 profilers' /metrics scraping ~100x and see if
# per-call latency drops (i.e. whether the profilers themselves were starving the server).
echo "=== launching $N_AGENTS concurrent '$TASK' agents  (hw_interval=${HW_INTERVAL:-0.05}s)  ($(date +%H:%M:%S)) ==="
python scripts/run_concurrent.py "$CFG" --base-url "http://127.0.0.1:$PORT/v1" --hw-interval "${HW_INTERVAL:-0.05}"
RC=$?
echo "=== run_concurrent.py exited rc=$RC ($(date)) ==="

# ---- teardown ----
kill "$HE_PID" "$SERVER_PID" 2>/dev/null || true
echo "=== job end $(date) ==="
echo "=== results in results_multi/  (session_comparison.txt = single-vs-multi table) ==="
