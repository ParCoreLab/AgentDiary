#!/bin/bash
# =============================================================================
# MN5 SUSTAINED-LOAD launcher — hold ~TARGET concurrent agents (closed loop, random task from a
# CPU-heavy mix, random arrivals) for a launch window, then drain. Stresses GPU serving AND the CPU
# tool phases at once, to see whether the CPU finally binds on the 32B model.
#
#     sbatch mn5_sustained.sh <target_concurrent> [launch_duration_s]
#   e.g.  sbatch mn5_sustained.sh 100 900     (hold ~100 concurrent for 15 min, then drain)
#         sbatch mn5_sustained.sh 40  600     (gentler validation run first)
# Watch:  jtail    Result: results_multi/session_<ts>/session_hardware.csv + session_sglang_metrics.csv
# =============================================================================
#SBATCH --job-name=biomni-sustained
#SBATCH --account=etur02
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/projects/etur02/koc858886/biomni/sustained_%j.log
#SBATCH --error=/gpfs/projects/etur02/koc858886/biomni/sustained_%j.log

set -uo pipefail
TARGET="${1:-100}"
DURATION="${2:-900}"

# RUNAWAY CAPS (evidence-based, updated 2026-08-08 after dissecting the 40×/100× runaways):
#  - MAX_CONSEC_MAXLEN=3: the real fix. Generate-runaways emit max-length (8192-tok) completions in a
#    loop and NEVER execute a tool; this aborts after 3 consecutive maxed completions w/ 0 tools run —
#    catches them at call ~4 (~800s) instead of wasting 1800s. Zero false-positives (healthy max consec=1).
#  - MAX_LLM_CALLS=20: legacy backstop (runaways rarely reach it — they need only ~9 giant calls).
#  - MAX_WALL_SECONDS=1200: backstop for the OTHER failure mode (a tool phase ballooning under contention);
#    lowered 1800→1200 now that the guard handles generate-runaways early.
export MAX_CONSEC_MAXLEN="${MAX_CONSEC_MAXLEN:-3}"
export MAXLEN_TOKENS="${MAXLEN_TOKENS:-8000}"
export MAX_LLM_CALLS="${MAX_LLM_CALLS:-20}"
export MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-1200}"

BASE=/gpfs/projects/etur02/koc858886/biomni
REPO=$BASE/biomni-profiling
VENV=$BASE/venv
PORT=30000
SERVER_LOG=$BASE/server_${SLURM_JOB_ID}.log
# 4 top CPU-demanding tasks (execute time / cores; archived fake tasks that hammer the CPU on purpose)
TASKS="tasks/_deprecated_fake/gsea_permutation.json \
       tasks/_deprecated_fake/betweenness_2k_network.json \
       tasks/_deprecated_fake/random_forest_cv_large.json \
       tasks/_deprecated_fake/spectral_clustering_3k.json"

echo "=== SUSTAINED job $SLURM_JOB_ID on $(hostname) | $(date) ==="
echo "=== target=$TARGET  launch_duration=${DURATION}s  MAX_LLM_CALLS=$MAX_LLM_CALLS  MAX_WALL_SECONDS=${MAX_WALL_SECONDS}s ==="
echo "=== tasks: $TASKS ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
module load anaconda/2024.02
module load cuda/12.6
source "$VENV/bin/activate"
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export DCGM_ROOT=$BASE/dcgm
export LD_LIBRARY_PATH="$DCGM_ROOT/usr/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$DCGM_ROOT/usr/bin:$PATH"
export DCGM_BINDINGS_PATH="$DCGM_ROOT/usr/share/datacenter-gpu-manager-4/bindings/python3"
MODEL="RyanLi0802/Biomni-R0-Preview"

echo "=== [setup] starting SGLang server ==="
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

echo "=== [setup] starting DCGM hostengine ==="
nv-hostengine -n > "$BASE/hostengine_${SLURM_JOB_ID}.log" 2>&1 &
HE_PID=$!
sleep 5
kill -0 "$HE_PID" 2>/dev/null && echo "  DCGM up (pid $HE_PID)" || echo "  DCGM failed -> NVML fallback"

cd "$REPO"
echo "=== launching sustained load (~$TARGET concurrent for ${DURATION}s, then drain) ==="
python scripts/run_sustained.py --target "$TARGET" --duration "$DURATION" --tasks $TASKS \
    --base-url "http://127.0.0.1:$PORT/v1"
echo "=== run_sustained.py exited rc=$? ($(date)) ==="

kill "$HE_PID" "$SERVER_PID" 2>/dev/null || true
echo "=== job end $(date) — results in results_multi/ (session_hardware.csv + session_sglang_metrics.csv) ==="
