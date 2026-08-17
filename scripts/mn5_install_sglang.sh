#!/bin/bash
# Installs SGLang into the shared MN5 venv. Runs on glogin1 (internet via du04 tunnel).
# Logs every step; safe to re-run. Watch with: tail -f install.log
set -o pipefail
VENV=/gpfs/projects/etur02/koc858886/biomni/venv

echo "=================================================================="
echo "=== SGLang install started: $(date) ==="
echo "=================================================================="

module load anaconda/2024.02
module load cuda/12.6
export https_proxy=socks5h://localhost:18080
export http_proxy=socks5h://localhost:18080
export HF_HUB_DISABLE_XET=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

source "$VENV/bin/activate"
echo "python: $(which python) $(python --version 2>&1)"
echo "nvcc:   $(which nvcc 2>/dev/null || echo none)"

echo ""
echo "=== [1/2] upgrading pip ==="
python -m pip install --upgrade pip 2>&1
echo "pip now: $(python -m pip --version)"

echo ""
echo "=== [2/2] installing sglang[all]==0.5.13.post1  (the big download, ~2-3 GB) ==="
python -m pip install "sglang[all]==0.5.13.post1" 2>&1
RC=$?

echo ""
echo "=== install finished rc=$RC at $(date) ==="
if [ "$RC" -eq 0 ]; then
  echo "=== quick import check ==="
  python -c "import sglang; print('OK sglang', sglang.__version__)" 2>&1
  python -c "import torch; print('OK torch', torch.__version__, 'cuda-build', torch.version.cuda)" 2>&1
fi
echo "=== ALL_DONE rc=$RC ==="
