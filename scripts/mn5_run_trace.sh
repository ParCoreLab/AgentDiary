#!/bin/bash
# =============================================================================
# MN5 launcher for a SINGLE-agent trace. Does what you did by hand on du04:
# reserve a GPU node -> start the SGLang server -> run the SAME trace_run.py
# against it -> shut the server down. Contains NO experiment logic.
#
# Usage (submit from an ACC login node, alogin1/alogin2):
#     sbatch mn5_run_trace.sh <task_id>
#     e.g.  sbatch mn5_run_trace.sh gsea_permutation
# Watch:  tail -f trace_<JOBID>.log
# =============================================================================
#SBATCH --job-name=biomni-trace
#SBATCH --account=etur02
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/etur02/koc858886/biomni/trace_%j.log
#SBATCH --error=/gpfs/projects/etur02/koc858886/biomni/trace_%j.log

set -uo pipefail
TASK="${1:?usage: sbatch mn5_run_trace.sh <task_id>   (e.g. gsea_permutation)}"
BASE=/gpfs/projects/etur02/koc858886/biomni
REPO=$BASE/biomni-profiling
VENV=$BASE/venv
PORT=30000
SERVER_LOG=$BASE/server_${SLURM_JOB_ID}.log

echo "=== node $(hostname) | job $SLURM_JOB_ID | task $TASK | $(date) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

module load anaconda/2024.02
module load cuda/12.6
source "$VENV/bin/activate"

export HF_HOME=$BASE/hf_cache          # the model we copied
export HF_HUB_OFFLINE=1                # compute node is air-gapped
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1

# --- userspace DCGM (RHEL9 build in /gpfs) so Layer-4 gets real SM-act/occ + power,
#     not just the NVML fallback. Needs LD_LIBRARY_PATH + the bindings path + a hostengine. ---
export DCGM_ROOT=$BASE/dcgm
export LD_LIBRARY_PATH="$DCGM_ROOT/usr/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$DCGM_ROOT/usr/bin:$PATH"
export DCGM_BINDINGS_PATH="$DCGM_ROOT/usr/share/datacenter-gpu-manager-4/bindings/python3"

MODEL="RyanLi0802/Biomni-R0-Preview"

# --- 1. start the SGLang server (same flags as du04's serve_biomni_r0.sh) ---
echo "=== starting SGLang server (loads ~122GB as bf16, a few minutes) ==="
python -m sglang.launch_server \
    --model-path "$MODEL" --port "$PORT" --host 127.0.0.1 \
    --tp 4 --dtype bfloat16 --mem-fraction-static 0.85 \
    --trust-remote-code --enable-metrics \
    --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":1.0,"original_max_position_embeddings":32768}, "max_position_embeddings": 131072}' \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# --- 2. wait until the server answers /health (up to ~20 min) ---
echo "=== waiting for server to be ready ==="
READY=0
for i in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "!!! server exited early:"; tail -n 40 "$SERVER_LOG"; break; fi
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ]; then
    READY=1; echo "=== server ready after ~$((i*10))s ==="; break
  fi
  sleep 10
done

# --- 3. run the SAME du04 trace_run.py (unchanged) ---
if [ "$READY" -eq 1 ]; then
  # start a private DCGM hostengine (-n foreground bg job; a detached daemon gets
  # killed by SLURM's cgroup). trace_run.py's Layer-4 connects to it via localhost.
  echo "=== starting userspace nv-hostengine for DCGM profiling ==="
  nv-hostengine -n > "$BASE/hostengine_${SLURM_JOB_ID}.log" 2>&1 &
  HE_PID=$!
  sleep 5
  if kill -0 "$HE_PID" 2>/dev/null; then echo "  DCGM hostengine up (pid $HE_PID)"; else echo "  DCGM hostengine failed -> Layer-4 will use NVML fallback"; fi
  cd "$REPO"
  echo "=== running trace_run.py for task: $TASK ==="
  python profiling/trace_run.py --task-config "tasks/${TASK}.json"
  echo "=== trace_run.py finished rc=$? ==="
  kill "$HE_PID" 2>/dev/null || true
  echo "=== results under: $REPO/results/${TASK}/ ==="
else
  echo "=== SERVER NEVER READY — see $SERVER_LOG ==="
fi

# --- 4. release the server / GPU ---
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "=== job end $(date) ==="
