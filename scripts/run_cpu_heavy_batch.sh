#!/bin/bash
# Run all CPU-heavy tasks sequentially and analyze each one.
# Usage: bash scripts/run_cpu_heavy_batch.sh
#
# Tasks ordered fastest → slowest; stop early and still have a spread.
# Each task auto-analyzes and updates aggregate.csv on completion.
#
# Task IDs reflect current graph/sample sizes (some filenames are legacy):
#   betweenness_1500_network.json → task_id: betweenness_5k_network
#   betweenness_2k_network.json   → task_id: betweenness_8k_network
#   spectral_clustering_3k.json   → task_id: spectral_clustering_8k
#   svm_rbf_8k.json               → task_id: svm_rbf_20k
#
# Prerequisites:
#   - SGLang server already running (scripts/serve_biomni_r0.sh)
#   - conda activate biomni-sglang
#   - Run from ~/biomni-profiling

set -e
cd "$(dirname "$0")/.."

LOG_DIR="results/batch_runs"
mkdir -p "$LOG_DIR"
BATCH_LOG="$LOG_DIR/batch_$(date +%Y-%m-%d_%H-%M-%S).log"

run_task() {
    local config="$1"
    local task_id
    task_id=$(python -c "import json; d=json.load(open('$config')); print(d['task_id'])")

    echo "========================================" | tee -a "$BATCH_LOG"
    echo "START: $task_id  $(date)" | tee -a "$BATCH_LOG"
    echo "========================================" | tee -a "$BATCH_LOG"

    python profiling/trace_run.py --task-config "$config" 2>&1 | tee -a "$BATCH_LOG"

    # Find the most recent result folder for this task
    latest=$(ls -dt "results/$task_id"/*/ 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "Analyzing: $latest" | tee -a "$BATCH_LOG"
        python profiling/analyze_trace.py "$latest" 2>&1 | tee -a "$BATCH_LOG"
    else
        echo "WARNING: no result folder found for $task_id" | tee -a "$BATCH_LOG"
    fi

    python profiling/aggregate_traces.py results/ 2>&1 | tee -a "$BATCH_LOG"
    echo "DONE: $task_id  $(date)" | tee -a "$BATCH_LOG"
    echo "" | tee -a "$BATCH_LOG"
}

# --- MODERATE bubble tasks (~30-50%), execute ~1-3 min each ---
run_task tasks/tsne_5k_synthetic.json          # tsne_5k_synthetic       ~30s exec
run_task tasks/spectral_clustering_3k.json     # spectral_clustering_8k  ~3-8 min exec (UPDATED)
run_task tasks/betweenness_3k_network.json     # betweenness_3k_network  ~30s exec

# --- HEAVY bubble tasks (~50-70%), execute ~3-8 min each ---
run_task tasks/gsea_permutation.json           # gsea_permutation        ~45s exec/turn, 1-2 turns (FIXED)
run_task tasks/random_forest_cv_large.json     # random_forest_cv_large  ~200s exec
run_task tasks/tsne_10k_synthetic.json         # tsne_10k_synthetic      ~33s exec
run_task tasks/tsne_multiscale_8k.json         # tsne_multiscale_8k      ~64s exec
run_task tasks/umap_15k_synthetic.json         # umap_15k_synthetic      ~44s exec
run_task tasks/betweenness_1500_network.json   # betweenness_5k_network  ~200s exec (UPDATED)

# --- EXTREME bubble tasks (~65-80%), execute 5-60 min each ---
run_task tasks/tsne_30k_synthetic.json         # tsne_30k_synthetic      ~95s exec
run_task tasks/umap_50k_synthetic.json         # umap_50k_synthetic      ~79s exec
run_task tasks/betweenness_2k_network.json     # betweenness_8k_network  ~250-400s exec (UPDATED)
run_task tasks/svm_rbf_8k.json                 # svm_rbf_20k             ~5-30 min exec (UPDATED)

echo "All tasks complete. Batch log: $BATCH_LOG"
echo "Aggregate results: results/aggregate.csv"
