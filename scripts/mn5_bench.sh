#!/bin/bash
# =============================================================================
# STAGE-2 raw-load benchmark (method D) — starts the SGLang server ONCE, then
# fires raw concurrent completion requests (NO Biomni) to characterize the
# server's true concurrent behavior and localize the 34s-TTFT cause.
#
#     sbatch --qos=acc_debug --time=02:00:00 mn5_bench.sh [sweep|probe]
#   sweep (default) = raw concurrency x prompt-size sweep (Stage 2)
#   probe           = 3a: fresh-request TTFT while N sustained long decodes run
# Watch:  tail -f bench_<JOBID>.log     Result: results_bench/bench_concurrent[_probe].json
# =============================================================================
#SBATCH --job-name=biomni-bench
#SBATCH --account=etur02
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/projects/etur02/koc858886/biomni/bench_%j.log
#SBATCH --error=/gpfs/projects/etur02/koc858886/biomni/bench_%j.log

set -uo pipefail
MODE="${1:-sweep}"
BASE=/gpfs/projects/etur02/koc858886/biomni
REPO=$BASE/biomni-profiling
VENV=$BASE/venv
PORT=30000
SERVER_LOG=$BASE/server_${SLURM_JOB_ID}.log

echo "=== BENCH job $SLURM_JOB_ID on $(hostname) | $(date) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
module load anaconda/2024.02
module load cuda/12.6
source "$VENV/bin/activate"
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
MODEL="RyanLi0802/Biomni-R0-Preview"

echo "=== starting SGLang server (same flags as all our runs) ==="
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

# log the server's effective scheduler config (max_running_requests, chunked-prefill, etc.)
echo "=== SGLang effective config (grep from server log) ==="
grep -iE "max_running_requests|chunked_prefill|max_prefill|schedule_policy|max_total_num_tokens|mem_fraction" "$SERVER_LOG" | head -20

cd "$REPO"
echo "=== running bench_concurrent.py (mode=$MODE) ==="
python profiling/bench_concurrent.py --model "$MODEL" --max-tokens 200 --mode "$MODE" --out "$REPO/results_bench/bench_concurrent.json"

kill "$SERVER_PID" 2>/dev/null || true
echo "=== bench done $(date) — results in results_bench/bench_concurrent.json ==="
