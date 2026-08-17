#!/usr/bin/env python3
"""
GPU-TOOL FEASIBILITY CHECK. Question: does Biomni's ADMET tool actually use the GPU, and how big is its
footprint -- i.e. could an "ADMET-screen a compound library" task be a REAL gpu-tool that contends with the
32B SGLang inference for HBM? (The old admet_ibuprofen was 1 compound = a millisecond MPNN pass, so it came
out generate-dominated / 3.3% bubble -- NOT a GPU load. We need a BATCH to know if it's usable.)

Run on du04 with the SGLang server DOWN (so the GPU is free to measure a clean footprint).
This needs the GPU but NOT the server or an agent launch.

    conda activate biomni-sglang
    python scripts/admet_gpu_footprint.py [N]        # N compounds, default 500

NOTE: DeepPurpose downloads pretrained models on first run -> needs internet (du04 ok, MN5 has none).
"""
import sys, time, subprocess

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def gpu_mem_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        return sum(int(x) for x in out.split())
    except Exception:
        return -1


import torch
cuda = torch.cuda.is_available()
drugs = [
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # caffeine
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "OC(=O)C1=CC=CC=C1",              # benzoic acid
]
smiles = (drugs * (N // len(drugs) + 1))[:N]

print(f"CUDA available to this process: {cuda}")
base = gpu_mem_used_mb()
if cuda:
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
from biomni.tool.pharmacology import predict_admet_properties
res = predict_admet_properties(smiles, ADMET_model_type="MPNN")
dt = time.time() - t0
after = gpu_mem_used_mb()
proc_peak = torch.cuda.max_memory_allocated() / 1e6 if cuda else 0.0

print(f"\nADMET(MPNN) on {N} compounds: {dt:.1f}s ({N/dt:.0f}/s)" if dt > 0 else "")
print(f"whole-card GPU mem (nvidia-smi): {base} MB -> {after} MB  (delta {after-base} MB)")
print(f"this-process GPU mem (torch peak): {proc_peak:.0f} MB")
gpu_used = (proc_peak > 200) or (after - base > 200)
print(f"VERDICT: {'GPU-TOOL (meaningful GPU load) -> viable as a gpu-tool task' if gpu_used else 'runs on CPU / trivial GPU -> NOT a usable gpu-tool'}")
print("\nCo-existence math: SGLang holds ~16GB weights + KV per card. The ADMET footprint above must fit in")
print("the leftover HBM (A100-40GB: ~24GB free after weights; H100-64GB: ~48GB free) for a gpu-tool to run")
print("concurrently with inference.")
print(f"\nresult head: {str(res)[:200]}")
