#!/bin/bash
# =============================================================================
# MN5 SWEEP launcher — load the server ONCE, run task(s) N times each against it.
# Purpose: measure the DISTRIBUTION of bubble/execute/generate. LLM generation
# length is stochastic (a single trace is misleading), so we repeat and aggregate.
# The ~7-min model load is paid once for the whole sweep.
#
# Submit from an ACC login node (alogin1/alogin2):
#     sbatch mn5_sweep.sh <n_reps> <task_id> [task_id2 ...]
#   e.g.  sbatch mn5_sweep.sh 100 gsea_permutation
#         sbatch mn5_sweep.sh 30 gsea_permutation gillespie_microbiome
#   quick test (fast 2h queue, 3 reps):
#         sbatch --qos=acc_debug --time=02:00:00 mn5_sweep.sh 3 gsea_permutation
# Watch:  tail -f sweep_<JOBID>.log
# Result: results/aggregate.csv (one row per run) = the distribution.
# =============================================================================
#SBATCH --job-name=biomni-sweep
#SBATCH --account=etur02
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/projects/etur02/koc858886/biomni/sweep_%j.log
#SBATCH --error=/gpfs/projects/etur02/koc858886/biomni/sweep_%j.log

set -uo pipefail
N_REPS="${1:?usage: sbatch mn5_sweep.sh <n_reps> <task_id> [task_id2 ...]}"
shift
TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && { echo "ERROR: no task_id given"; exit 1; }
# Runaway handling — two layers:
#  1) MAX_WALL_SECONDS (in-run, GRACEFUL): trace_run.py aborts a run that exceeds it at the next
#     LLM call and RECORDS it as agent_completed=False, so every failure appears in aggregate.csv.
#     gsea completes in ~4-9 min (incl. occasional double-execute ~520s); 720s clears that.
#  2) CAP_SECONDS (external SIGKILL): last-resort backstop for a true hang (e.g. a stuck execute
#     that never reaches an LLM call). Set high — it should rarely fire, and a run it kills is
#     NOT recorded (visible only in this log). Tune either via --export=ALL,VAR=val.
export MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-720}"
CAP_SECONDS="${CAP_SECONDS:-1800}"

BASE=/gpfs/projects/etur02/koc858886/biomni
REPO=$BASE/biomni-profiling
VENV=$BASE/venv
PORT=30000
SERVER_LOG=$BASE/server_${SLURM_JOB_ID}.log

echo "=== SWEEP job $SLURM_JOB_ID on $(hostname) | $(date) ==="
echo "=== reps=$N_REPS  graceful-wall-cap=${MAX_WALL_SECONDS}s  backstop-kill=${CAP_SECONDS}s  tasks: ${TASKS[*]} ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

module load anaconda/2024.02
module load cuda/12.6
source "$VENV/bin/activate"
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
# userspace DCGM (real SM-act/occ + power)
export DCGM_ROOT=$BASE/dcgm
export LD_LIBRARY_PATH="$DCGM_ROOT/usr/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$DCGM_ROOT/usr/bin:$PATH"
export DCGM_BINDINGS_PATH="$DCGM_ROOT/usr/share/datacenter-gpu-manager-4/bindings/python3"
MODEL="RyanLi0802/Biomni-R0-Preview"

# ---- SETUP (once): SGLang server ----
echo "=== [setup] starting SGLang server (loads ~122GB, several min) ==="
python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --host 127.0.0.1 \
    --tp 4 --dtype bfloat16 --mem-fraction-static 0.85 --trust-remote-code --enable-metrics \
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

# ---- SETUP (once): DCGM hostengine ----
echo "=== [setup] starting DCGM hostengine ==="
nv-hostengine -n > "$BASE/hostengine_${SLURM_JOB_ID}.log" 2>&1 &
HE_PID=$!
sleep 5
if kill -0 "$HE_PID" 2>/dev/null; then echo "  DCGM hostengine up (pid $HE_PID)"; else echo "  DCGM failed -> Layer-4 uses NVML"; fi

# ---- SWEEP ----
cd "$REPO"
total=0; ok=0
for TASK in "${TASKS[@]}"; do
  if [ ! -f "tasks/${TASK}.json" ]; then echo "!! tasks/${TASK}.json not found — skipping"; continue; fi
  for rep in $(seq 1 "$N_REPS"); do
    total=$((total+1))
    echo ""
    echo "########## $TASK  rep $rep/$N_REPS  ($(date +%H:%M:%S)) ##########"
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "!! server died mid-sweep — aborting"; break 2; fi
    if timeout --signal=TERM --kill-after=30s "$CAP_SECONDS" python profiling/trace_run.py --task-config "tasks/${TASK}.json"; then
      ok=$((ok+1))   # trace RECORDED — a completion OR a gracefully-capped failure; split by agent_completed
    else
      rc=$?
      if [ "$rc" = "124" ]; then echo "!! rep hit the ${CAP_SECONDS}s BACKSTOP kill — NOT recorded (should be rare)"
      else echo "!! rep crashed (rc=$rc) — continuing"; fi
    fi
  done
done
echo ""
echo "=== sweep complete: $ok/$total reps recorded to aggregate; $((total-ok)) hit the backstop kill (not recorded, should be 0) ==="
echo "=== TRUE completion vs failure split = the agent_completed column in results/aggregate.csv ==="

# ---- AGGREGATE the distribution ----
echo "=== building results/aggregate.csv (one row per run) ==="
python profiling/aggregate_traces.py results/ 2>&1 | tail -3
echo "=== aggregate.csv at: $REPO/results/aggregate.csv ==="

# ---- TEARDOWN ----
kill "$HE_PID" "$SERVER_PID" 2>/dev/null || true
echo "=== job end $(date) ==="
