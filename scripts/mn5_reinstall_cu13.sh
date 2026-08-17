#!/bin/bash
# Rebuild the MN5 SGLang env with the ORIGINAL coherent CUDA-13 stack.
# WHY: MN5 compute nodes have driver 595 (CUDA-13 capable, seen in the serve test),
# so cu130 works here AND matches du04 exactly. This undoes the CUDA-12.9 detour
# that caused the sgl_kernel version-skew (missing fp8_blockwise_scaled_mm on H100).
# Run on glogin1 (internet via the du04 tunnel).  Watch: tail -f reinstall_cu13.log
set -o pipefail
BASE=/gpfs/projects/etur02/koc858886/biomni
VENV=$BASE/venv

echo "=================================================================="
echo "=== fresh CUDA-13 SGLang install: $(date) ==="
echo "=================================================================="
module load anaconda/2024.02
export https_proxy=socks5h://localhost:18080
export http_proxy=socks5h://localhost:18080
export HF_HUB_DISABLE_XET=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Rebuild the venv from scratch (the old one has the broken cu129 mix).
echo "=== recreating venv at $VENV ==="
rm -rf "$VENV"
python3 -m venv "$VENV"

# Bootstrap PySocks into the venv using the BASE anaconda python (which HAS PySocks),
# BEFORE activating the venv — otherwise the venv's own fresh pip can't use the SOCKS
# tunnel yet, and every download fails with "Missing dependencies for SOCKS support".
echo "=== bootstrapping PySocks (base anaconda python -> venv site-packages) ==="
python3 -m pip install --no-cache-dir --target="$VENV/lib/python3.11/site-packages" PySocks

# Now the venv has PySocks, so its own pip can use the tunnel.
source "$VENV/bin/activate"

# Do NOT use pip 26.x: it has a SOCKS-proxy regression
# ("PoolKey.__new__() got an unexpected keyword argument 'key_proxy_ssl_context'").
# Pin to 24.3.1 — SOCKS-safe and new enough for modern wheel metadata.
echo "=== pinning pip to 24.3.1 (26.x breaks SOCKS) ==="
python -m pip install "pip==24.3.1"

echo "=== installing sglang[all]==0.5.13.post1  (default CUDA-13, coherent set — ~30-40 min) ==="
python -m pip install "sglang[all]==0.5.13.post1"
RC=$?

echo ""
echo "=== installed versions ==="
python -c "import sglang, torch; print('sglang', sglang.__version__, '| torch', torch.__version__, '| cuda-build', torch.version.cuda)" 2>&1
python -c "import sgl_kernel; print('sgl_kernel', getattr(sgl_kernel,'__version__','?'))" 2>&1
echo "=== DONE rc=$RC $(date) ==="
