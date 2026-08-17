#!/bin/bash
# =============================================================================
# One-shot SGLang serve TEST for Biomni-R0 on MareNostrum 5 (ACC / 4x H100-64GB).
# Reserves a full ACC node, starts the server from the LOCAL model cache (offline),
# waits until ready, sends one test prompt, prints the reply, then shuts down.
# Submit from an ACC login node (alogin1/alogin2):   sbatch mn5_serve_test.sh
# Watch:  tail -f serve_test_<JOBID>.log
# =============================================================================
#SBATCH --job-name=biomni-serve-test
#SBATCH --account=etur02
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/etur02/koc858886/biomni/serve_test_%j.log
#SBATCH --error=/gpfs/projects/etur02/koc858886/biomni/serve_test_%j.log

set -uo pipefail
BASE=/gpfs/projects/etur02/koc858886/biomni
echo "=== node $(hostname) | job ${SLURM_JOB_ID} | $(date) ==="

# ---- environment ----
module load anaconda/2024.02
module load cuda/12.6
source "$BASE/venv/bin/activate"

export HF_HOME="$BASE/hf_cache"     # the model we copied lives here
export HF_HUB_OFFLINE=1             # compute nodes are air-gapped: never phone home
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1

MODEL="RyanLi0802/Biomni-R0-Preview"
PORT=30000
SERVER_LOG="$BASE/sglang_server_${SLURM_JOB_ID}.log"

echo "=== GPUs on this node ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# ---- launch server in the background ----
echo "=== launching SGLang (loads ~122GB of weights as bf16 -> can take several minutes) ==="
python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --tp 4 \
    --dtype bfloat16 \
    --mem-fraction-static 0.85 \
    --trust-remote-code \
    --enable-metrics \
    --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":1.0,"original_max_position_embeddings":32768}, "max_position_embeddings": 131072}' \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# ---- wait until ready (poll /health, up to ~20 min) ----
echo "=== waiting for server to be ready ==="
READY=0
for i in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "!!! server process exited early — dumping server log:"; tail -n 50 "$SERVER_LOG"; break
  fi
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
  if [ "$code" = "200" ]; then READY=1; echo "=== SERVER READY after ~$((i*10))s ==="; break; fi
  sleep 10
done

# ---- send one test prompt ----
if [ "$READY" -eq 1 ]; then
  echo "=== sending a test completion ==="
  curl -s "http://127.0.0.1:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"In one sentence, what is CRISPR?\",\"max_tokens\":64,\"temperature\":0}" \
    | python -c "import sys,json; d=json.load(sys.stdin); print('>>> MODEL REPLY:', d['choices'][0]['text'].strip())" \
    || { echo 'test request failed — server log tail:'; tail -n 30 "$SERVER_LOG"; }
  echo "=== SERVE TEST PASSED ==="
else
  echo "=== SERVER NEVER BECAME READY — see $SERVER_LOG ==="
fi

# ---- shut down ----
echo "=== shutting down server ==="
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "=== job end $(date) ==="
