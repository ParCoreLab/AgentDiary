#!/bin/bash
# =============================================================================
# MN5 CPU-THROTTLE experiment — CAUSAL proof that CPU (not GPU) bounds the tool phases.
# Hold offered concurrency FIXED, vary the number of CPU cores the AGENTS may use, and measure
# whether execute (tool) time inflates as cores shrink WHILE GPU serving (KV/queue/TTFT/TPOT) stays flat.
# If execute ∝ 1/cores and GPU metrics don't move → the CPU is the bottleneck, nothing else. QED.
#
# Design (rigorous): the SGLang server is pinned to its own reserved cores (0..RESERVE-1) so it is NEVER
# throttled — only the agent tool phases are. The agent launcher (run_sustained.py) is pinned to the next
# CORES cores; its subprocesses (the agents) inherit that CPU affinity, so all tool computation is confined
# to exactly CORES logical cores. Each run lands under results_multi/throttle_<CORES>c/ (self-labeling).
#
#   sbatch --qos=acc_ehpc --time=02:00:00 mn5_cpu_throttle.sh <CORES> [TARGET] [DURATION]
# Recommended sweep at TARGET=40 (full-node demand ~92 cores, so this spans UNBOUND→BOUND):
#   sbatch --qos=acc_ehpc --time=02:00:00 mn5_cpu_throttle.sh 120     # unbound baseline (92<120)
#   sbatch --qos=acc_ehpc --time=02:00:00 mn5_cpu_throttle.sh 72      # binds        (92>72)
#   sbatch --qos=acc_ehpc --time=02:00:00 mn5_cpu_throttle.sh 40      # hard bound   (92>>40)
# =============================================================================
#SBATCH --job-name=biomni-throttle
#SBATCH --account=etur02
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/etur02/koc858886/biomni/throttle_%j.log
#SBATCH --error=/gpfs/projects/etur02/koc858886/biomni/throttle_%j.log

set -uo pipefail
CORES="${1:?usage: sbatch mn5_cpu_throttle.sh <CORES> [TARGET=40] [DURATION=500]}"
TARGET="${2:-40}"
DURATION="${3:-500}"
RESERVE=24                         # logical cores reserved for the SGLang server (never throttled)
AGENT_LO=$RESERVE
AGENT_HI=$((RESERVE + CORES - 1))
if [ "$AGENT_HI" -gt 159 ]; then echo "ERROR: RESERVE($RESERVE)+CORES($CORES) exceeds 160 logical cores"; exit 1; fi

# Runaway guard ON (kills generate-loops at ~call 4) so a throttled run isn't polluted by runaways.
export MAX_CONSEC_MAXLEN="${MAX_CONSEC_MAXLEN:-3}"
export MAXLEN_TOKENS="${MAXLEN_TOKENS:-8000}"
export MAX_LLM_CALLS="${MAX_LLM_CALLS:-20}"
export MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-1200}"

BASE=/gpfs/projects/etur02/koc858886/biomni
REPO=$BASE/biomni-profiling
VENV=$BASE/venv
PORT=30000
SERVER_LOG=$BASE/server_${SLURM_JOB_ID}.log
TASKS="tasks/_deprecated_fake/gsea_permutation.json \
       tasks/_deprecated_fake/betweenness_2k_network.json \
       tasks/_deprecated_fake/random_forest_cv_large.json \
       tasks/_deprecated_fake/spectral_clustering_3k.json"

echo "=== CPU-THROTTLE job $SLURM_JOB_ID on $(hostname) | $(date) ==="
echo "=== CORES=$CORES (agents pinned $AGENT_LO-$AGENT_HI)  server pinned 0-$((RESERVE-1))  TARGET=$TARGET  DURATION=${DURATION}s ==="
echo "=== guard: MAX_CONSEC_MAXLEN=$MAX_CONSEC_MAXLEN  MAX_WALL_SECONDS=${MAX_WALL_SECONDS}s ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader
module load anaconda/2024.02
module load cuda/12.6
source "$VENV/bin/activate"
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export DCGM_ROOT=$BASE/dcgm
export LD_LIBRARY_PATH="$DCGM_ROOT/usr/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$DCGM_ROOT/usr/bin:$PATH"
export DCGM_BINDINGS_PATH="$DCGM_ROOT/usr/share/datacenter-gpu-manager-4/bindings/python3"
MODEL="RyanLi0802/Biomni-R0-Preview"

echo "=== [setup] starting SGLang server PINNED to cores 0-$((RESERVE-1)) ==="
taskset -c 0-$((RESERVE-1)) python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --host 127.0.0.1 \
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
HE_PID=$!; sleep 5
kill -0 "$HE_PID" 2>/dev/null && echo "  DCGM up" || echo "  DCGM failed -> NVML fallback"

cd "$REPO"
echo "=== launching agents PINNED to cores $AGENT_LO-$AGENT_HI ($CORES cores), target=$TARGET ==="
taskset -c $AGENT_LO-$AGENT_HI python scripts/run_sustained.py --target "$TARGET" --duration "$DURATION" \
    --tasks $TASKS --base-url "http://127.0.0.1:$PORT/v1" --results-root "results_multi/throttle_${CORES}c"
echo "=== run_sustained.py exited rc=$? ($(date)) ==="

kill "$HE_PID" "$SERVER_PID" 2>/dev/null || true
echo "=== job end $(date) — results in results_multi/throttle_${CORES}c/ ==="
