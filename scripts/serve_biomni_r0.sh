#!/usr/bin/env bash
# =============================================================================
# SGLang server launch for Biomni-R0 (the REAL 32B model) on du04.
# =============================================================================
# Long-lived server. Run in its own terminal (or tmux pane) in biomni-sglang:
#     conda activate biomni-sglang
#     bash serve_biomni_r0.sh
# Ready when you see: "The server is fired up and ready to roll!"
# Endpoint: http://localhost:30000/v1
#
# GPU SELECTION: all 4 A100-40GB cards (checked free via nvidia-smi). The 32B
#   model was downloaded as F32 (~125GB, 27 shards) and does NOT fit on fewer
#   cards. We use all four and do NOT pin CUDA_VISIBLE_DEVICES so SGLang sees
#   all of them. Re-check `nvidia-smi` before launching: du04 is shared.
#
# WHY EACH FLAG:
#   --model-path RyanLi0802/Biomni-R0-Preview : resolves from the local HF
#       cache (/DATA/mansari26/huggingface_cache). No download.
#   --tp 4 : tensor-parallel across all 4 cards. Required for this model size.
#   --dtype bfloat16 : load the F32 weights as bf16. Halves weight memory
#       (~125GB F32 -> ~63GB bf16, ~16GB/card), leaving ~24GB/card for the KV
#       cache and the long context R0 needs. No reasoning-quality cost here.
#   --mem-fraction-static 0.85 : reserve 85% of each card for weights + KV cache.
#   --trust-remote-code : R0 ships a custom model definition.
#   (CUDA_HOME is set above so nvcc is found; FlashInfer + CUDA graphs compile
#    normally -- no backend workarounds needed.)
#   rope yarn -> max_position_embeddings 131072 : extends native 32K context to
#       131K so Biomni's ~24-30K-token system prompt fits with room to spare.
# =============================================================================

set -euo pipefail

# du04 has the CUDA toolkit under /usr/local/cuda-13.0 but it is not on PATH
# and /usr/local/cuda doesn't exist, so SGLang/FlashInfer couldn't find nvcc.
# Point CUDA_HOME at the toolkit matching the driver (CUDA 13.0) and expose nvcc.
export CUDA_HOME="/usr/local/cuda-13.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export HF_HOME="/DATA/mansari26/huggingface_cache"
export HF_HUB_DISABLE_XET=1     # du04's network blocks the Xet protocol
export HF_XET_DISABLE=1

MODEL="RyanLi0802/Biomni-R0-Preview"
PORT=30000

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tp 4 \
    --dtype bfloat16 \
    --mem-fraction-static 0.85 \
    --trust-remote-code \
    --enable-metrics \
    --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":1.0,"original_max_position_embeddings":32768}, "max_position_embeddings": 131072}'