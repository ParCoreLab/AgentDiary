#!/bin/bash
# Install Biomni + its agent stack + scientific libs into the MN5 venv, matching du04.
# Biomni core is light (langchain/pydantic/dotenv); the scientific libs are what the TASKS use.
# All versions pinned to du04's biomni-sglang env so behaviour is identical, and NOTHING here
# should touch torch/sglang (verified at the end).  Run on glogin1 (internet via tunnel).
# Watch: tail -f install_biomni.log
set -o pipefail
BASE=/gpfs/projects/etur02/koc858886/biomni
VENV=$BASE/venv
BIOMNI_SRC=$BASE/Biomni           # the source clone copied from du04

echo "=== biomni install: $(date) ==="
module load anaconda/2024.02
export https_proxy=socks5h://localhost:18080
export http_proxy=socks5h://localhost:18080
export HF_HUB_DISABLE_XET=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
source "$VENV/bin/activate"

echo "=== torch/sglang BEFORE (must be identical AFTER) ==="
python -c "import torch,sglang; print('  torch',torch.__version__,'| sglang',sglang.__version__)"

echo ""
echo "=== [1/3] langchain agent stack (pinned to du04) ==="
python -m pip install \
  "langchain==1.3.9" "langchain-core==1.4.7" "langchain-openai==1.3.2" \
  "langgraph==1.2.5" "langsmith==0.8.16" "openai==2.43.0" \
  "pydantic==2.13.4" "python-dotenv==1.2.2" "tiktoken==0.13.0"

echo ""
echo "=== [2/3] scientific libs the tasks use (pinned to du04) ==="
python -m pip install \
  "numpy==2.3.5" "scipy==1.17.1" "pandas==2.3.3" "scikit-learn==1.9.0" \
  "networkx==3.6.1" "umap-learn==0.5.12" "numba==0.65.1" "llvmlite==0.47.0" \
  "joblib==1.5.3" "threadpoolctl==3.6.0" "pynndescent==0.6.0"

echo ""
echo "=== [3/3] biomni 0.0.8 from the copied source clone (no deps) ==="
python -m pip install -e "$BIOMNI_SRC" --no-deps

echo ""
echo "=== verify: everything imports AND torch/sglang UNCHANGED ==="
python -c "
import torch, sglang, numpy, scipy, sklearn, umap, networkx, pandas
import langchain, langgraph, langchain_openai
import biomni
from biomni.agent import A1
print('  OK  torch', torch.__version__, '| sglang', sglang.__version__, '| biomni', getattr(biomni,'__version__','?'))
print('  OK  langchain', langchain.__version__, '| A1 agent imported')
" 2>&1 | grep -vE "NVML|_raw_device_count"
echo "=== DONE $(date) ==="
