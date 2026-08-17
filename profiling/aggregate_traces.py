#!/usr/bin/env python3
"""
Aggregate analysis.json files across all trace run folders.

Usage:
    python profiling/aggregate_traces.py              # reads results/, writes results/aggregate.csv
    python profiling/aggregate_traces.py results/

For each folder under results_dir that contains an analysis.json, one row is
written to aggregate.csv.  Run analyze_trace.py on each folder first.
"""

import argparse
import csv
import json
from pathlib import Path


# Flat columns to extract from analysis.json → one row per trace.
# Each entry: (csv_column_name, extractor_function).
# Extractors receive the full analysis dict; return None on missing data.
FIELDS = [
    # identity
    ("trace_id",               lambda a: a["trace_id"]),
    ("date",                   lambda a: a["meta"]["date"]),
    ("task_id",                lambda a: a["meta"].get("task_id")),
    ("task_family",            lambda a: a["meta"].get("task_family")),
    ("resource_profile_expected", lambda a: a["meta"].get("resource_profile_expected")),
    ("task",                   lambda a: (a["meta"].get("task") or "")[:100]),
    ("model",                  lambda a: a["meta"].get("model")),
    ("error",                  lambda a: a["meta"].get("error")),
    # agent_completed = the run actually executed its tool AND had no error. False marks
    # runaway/format-failure runs (model never emitted a code block -> execute_total 0 or an
    # error) so the bubble distribution can be computed over completed runs only, while the
    # failures are still counted (the failure rate is itself a result).
    ("agent_completed",        lambda a: (a["meta"].get("error") is None) and ((a.get("timing", {}).get("execute_total_s") or 0) > 5)),
    # top-level time
    ("wall_time_s",            lambda a: a["meta"]["wall_time_s"]),
    ("n_hw_samples",           lambda a: a["meta"]["n_hw_samples"]),
    # turn structure
    ("n_generate_spans",       lambda a: a["turns"]["n_generate_spans"]),
    ("n_execute_spans",        lambda a: a["turns"]["n_execute_spans"]),
    ("n_pre_loop_generates",   lambda a: a["turns"]["n_pre_loop_generates"]),
    ("n_real_turns",           lambda a: a["turns"]["n_real_turns"]),
    # timing
    ("init_time_s",            lambda a: a["timing"]["init_time_s"]),
    ("agent_active_s",         lambda a: a["timing"]["agent_active_s"]),
    ("generate_total_s",       lambda a: a["timing"]["generate_total_s"]),
    ("execute_total_s",        lambda a: a["timing"]["execute_total_s"]),
    ("inter_phase_s",          lambda a: a["timing"]["inter_phase_s"]),
    ("generate_mean_s",        lambda a: a["timing"]["generate_mean_s"]),
    ("generate_std_s",         lambda a: a["timing"]["generate_std_s"]),
    ("execute_mean_s",         lambda a: a["timing"]["execute_mean_s"]),
    ("execute_std_s",          lambda a: a["timing"]["execute_std_s"]),
    # bubble — primary metrics
    ("gpu_bubble_weighted_s",     lambda a: a["bubble"]["gpu_weighted_s"]),
    ("gpu_bubble_weighted_frac",  lambda a: a["bubble"]["gpu_weighted_fraction"]),
    ("gpu_bubble_binary_s",       lambda a: a["bubble"]["gpu_binary_s"]),
    ("gpu_bubble_binary_frac",    lambda a: a["bubble"]["gpu_binary_fraction"]),
    ("cpu_bubble_weighted_s",     lambda a: a["bubble"]["cpu_weighted_s"]),
    ("cpu_bubble_weighted_frac",  lambda a: a["bubble"]["cpu_weighted_fraction"]),
    ("cpu_bubble_binary_s",       lambda a: a["bubble"]["cpu_binary_s"]),
    ("cpu_bubble_binary_frac",    lambda a: a["bubble"]["cpu_binary_fraction"]),
    ("layer1_gpu_bubble_frac",    lambda a: a["bubble"]["layer1_gpu_bubble_fraction"]),
    # GPU saturation (new detailed fields)
    ("gpu_sm_util_generate",      lambda a: a["gpu_saturation"]["during_generate"]["sm_util_mean"]),
    ("gpu_mem_bw_generate",       lambda a: a["gpu_saturation"]["during_generate"]["mem_bw_util_mean"]),
    ("gpu_mem_used_generate_mb",  lambda a: a["gpu_saturation"]["during_generate"]["mem_used_mean_mb"]),
    ("gpu_mem_free_generate_mb",  lambda a: a["gpu_saturation"]["during_generate"]["mem_free_mean_mb"]),
    ("gpu_sm_util_execute",       lambda a: a["gpu_saturation"]["during_execute"]["sm_util_mean"]),
    ("gpu_mem_bw_execute",        lambda a: a["gpu_saturation"]["during_execute"]["mem_bw_util_mean"]),
    ("kv_cache_peak_pct",         lambda a: a["gpu_saturation"]["kv_cache"]["peak_pct"]),
    ("kv_cache_end_pct",          lambda a: a["gpu_saturation"]["kv_cache"]["end_pct"]),
    # CPU utilization
    ("cpu_mean_generate",         lambda a: a["cpu_utilization"]["during_generate"]["mean"]),
    ("cpu_max_generate",          lambda a: a["cpu_utilization"]["during_generate"]["max_core"]),
    ("cpu_hot_cores_generate",    lambda a: a["cpu_utilization"]["during_generate"]["hot_cores"]),
    ("cpu_mean_execute",          lambda a: a["cpu_utilization"]["during_execute"]["mean"]),
    ("cpu_max_execute",           lambda a: a["cpu_utilization"]["during_execute"]["max_core"]),
    ("cpu_hot_cores_execute",     lambda a: a["cpu_utilization"]["during_execute"]["hot_cores"]),
    ("peak_cpu_mean",             lambda a: a["cpu_utilization"]["peak_cpu_mean"]),
    ("peak_cpu_max",              lambda a: a["cpu_utilization"]["peak_cpu_max"]),
    # Layer-2: HTTP token counts (Gen-2 traces only; None for Gen-1)
    ("l2_n_llm_calls",           lambda a: a["layer2"]["n_llm_calls"]),
    ("l2_total_prompt_tokens",   lambda a: a["layer2"]["total_prompt_tokens"]),
    ("l2_total_completion_tokens", lambda a: a["layer2"]["total_completion_tokens"]),
    ("l2_mean_latency_s",        lambda a: a["layer2"]["mean_latency_s"]),
    ("l2_mean_prompt_tokens",    lambda a: a["layer2"]["mean_prompt_tokens"]),
    ("l2_mean_completion_tokens", lambda a: a["layer2"]["mean_completion_tokens"]),
    # Layer-3: SGLang engine metrics (Gen-2 traces only)
    ("l3_mean_ttft_s",           lambda a: a["layer3"]["mean_ttft_s"]),
    ("l3_mean_tpot_ms",          lambda a: a["layer3"]["mean_tpot_ms"]),
    ("l3_mean_queue_time_ms",    lambda a: a["layer3"]["mean_queue_time_ms"]),
    ("l3_queue_wait_mean_ms",    lambda a: a["layer3"]["queue_wait_mean_ms"]),
    ("l3_mean_engine_e2e_s",     lambda a: a["layer3"]["mean_engine_e2e_s"]),
    ("l3_fwd_extend_total_s",    lambda a: a["prefill_vs_decode"]["total_fwd_extend_s"]),
    ("l3_fwd_decode_total_s",    lambda a: a["prefill_vs_decode"]["total_fwd_decode_s"]),
    ("l3_extend_fraction",       lambda a: a["prefill_vs_decode"]["extend_fraction"]),
    # tool inventory (pipe-separated within the cell to avoid CSV quoting issues)
    ("execute_resource_types",    lambda a: "|".join(
                                      e["resource_type"] for e in a["execute_spans"])),
    ("n_trivial_executes",        lambda a: sum(1 for e in a["execute_spans"]
                                                if e["resource_type"] == "trivial")),
    ("n_gpu_tool_executes",       lambda a: sum(1 for e in a["execute_spans"]
                                                if e["resource_type"] == "gpu_tool")),
    ("n_cpu_tool_executes",       lambda a: sum(1 for e in a["execute_spans"]
                                                if e["resource_type"] == "cpu_tool")),
    ("n_network_executes",        lambda a: sum(1 for e in a["execute_spans"]
                                                if e["resource_type"] == "network_tool")),
    ("n_mixed_executes",          lambda a: sum(1 for e in a["execute_spans"]
                                                if e["resource_type"] == "mixed")),
]


def aggregate(results_dir: Path):
    # Scan recursively: works for both flat (results/{timestamp}/) and
    # nested (results/{task_id}/{timestamp}/) directory structures.
    analysis_files = sorted(results_dir.rglob("analysis.json"))
    folders = [f.parent for f in analysis_files]
    if not folders:
        print(f"No analysis.json found under {results_dir}/")
        print("Run:  python profiling/analyze_trace.py <trace_folder>  for each trace first.")
        return

    rows = []
    for folder in folders:
        a = json.loads((folder / "analysis.json").read_text())
        row = {}
        for col, extractor in FIELDS:
            try:
                row[col] = extractor(a)
            except (KeyError, TypeError, IndexError):
                row[col] = None
        rows.append(row)
        print(f"  loaded  {folder.name}")

    out = results_dir / "aggregate.csv"
    col_names = [col for col, _ in FIELDS]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=col_names)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote {len(rows)} trace(s) → {out}")
    print(f"Columns: {len(col_names)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Aggregate all trace analysis.json files into one CSV.")
    p.add_argument("results_dir", nargs="?", default="results",
                   help="results directory (default: results/)")
    args = p.parse_args()
    aggregate(Path(args.results_dir))
