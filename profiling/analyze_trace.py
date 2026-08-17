#!/usr/bin/env python3
"""
Analyze one trace run folder.

Usage:
    python profiling/analyze_trace.py results/scrna_pbmc3k/2026-06-23_11-46-09/
    python profiling/analyze_trace.py results/umap_large_synthetic/2026-06-23_12-39-23/ --no-plot

Outputs written into the same folder:
    analysis.json  — canonical per-trace metrics
    timeline.png   — phase timeline with GPU/CPU overlay

Handles two hardware.csv generations:
  Gen 1 (old trace_run.py): gpu{i}_util, gpu{i}_mem_mb (no BW, no free)
  Gen 2 (new trace_run.py): gpu{i}_sm_act, gpu{i}_sm_occ, gpu{i}_dram_act,
                             gpu{i}_power_w, gpu{i}_sm_clock_mhz,
                             gpu{i}_mem_used_mb, gpu{i}_mem_free_mb

Old analysis.json files (gpu_utilization section) are left untouched; this
script only overwrites when it re-analyzes a folder.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

N_GPUS = 4
GPU_SM_BUSY_THRESHOLD = 20   # % SM util below this during execute = GPU bubble
CPU_BUSY_THRESHOLD    = 20   # % cpu_mean below this during generate = CPU bubble
N_CPUS                = 256  # du04 logical cores


# ── data loading ──────────────────────────────────────────────────────────────

def load_events(folder: Path):
    """
    Parse events.jsonl.
    Returns (gen_spans, exec_spans, layer2_calls):
      gen_spans / exec_spans — sorted span dicts {t_start, t_end, duration_s, ...}
      layer2_calls — list of {seq, latency_s, prompt_tokens, completion_tokens}
                     extracted from generate:end events (present only in Gen-2 traces)
    """
    raw = [json.loads(l) for l in (folder / "events.jsonl").read_text().splitlines() if l.strip()]
    pending = {}
    gen_spans, exec_spans = [], []
    layer2_calls = []

    for e in raw:
        if e["edge"] == "start":
            pending[e["phase"]] = e
        elif e["edge"] == "end" and e["phase"] in pending:
            s = pending.pop(e["phase"])
            span = {
                "t_start":    s["t"],
                "t_end":      e["t"],
                "duration_s": round(e["t"] - s["t"], 6),
                "fn":   s.get("fn")   or e.get("fn"),
                "lang": s.get("lang") or e.get("lang"),
            }
            if e["phase"] == "generate":
                gen_spans.append(span)
                # Layer-2: token counts and end-to-end HTTP latency (Gen-2 only)
                if e.get("seq") is not None and e.get("latency_s") is not None:
                    layer2_calls.append({
                        "seq":               e["seq"],
                        "latency_s":         e.get("latency_s"),
                        "prompt_tokens":     e.get("prompt_tokens"),
                        "completion_tokens": e.get("completion_tokens"),
                        "total_tokens":      e.get("total_tokens"),
                    })
            else:
                exec_spans.append(span)

    return (sorted(gen_spans,  key=lambda x: x["t_start"]),
            sorted(exec_spans, key=lambda x: x["t_start"]),
            layer2_calls)


def load_hw(folder: Path):
    rows = []
    with open(folder / "hardware.csv") as f:
        for r in csv.DictReader(f):
            parsed = {}
            for k, v in r.items():
                try:
                    parsed[k] = float(v) if v != "" else None
                except ValueError:
                    parsed[k] = v   # keep strings (hw_source)
            rows.append(parsed)
    return rows


def load_sglang_metrics(folder: Path):
    """Load sglang_metrics.csv if present (Gen-2 traces only). Returns [] if missing."""
    p = folder / "sglang_metrics.csv"
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            parsed = {}
            for k, v in r.items():
                try:
                    parsed[k] = float(v) if v != "" else None
                except ValueError:
                    parsed[k] = None
            rows.append(parsed)
    return rows


def _detect_hw_format(hw):
    """
    Returns (sm_col, bw_col, mem_used_col, mem_free_col) — suffix strings for
    _gpu_mean() calls.  None means the column doesn't exist in this trace.

    Gen 1: gpu{i}_util     → sm_col="util",   bw_col=None,       mem_used="mem_mb",    mem_free=None
    Gen 2: gpu{i}_sm_act   → sm_col="sm_act", bw_col="dram_act", mem_used="mem_used_mb", mem_free="mem_free_mb"
    """
    if not hw:
        return "sm_act", "dram_act", "mem_used_mb", "mem_free_mb"
    row = hw[0]
    if "gpu0_sm_act" in row:
        return "sm_act", "dram_act", "mem_used_mb", "mem_free_mb"
    if "gpu0_util" in row:
        return "util", None, "mem_mb", None
    return "sm_act", "dram_act", "mem_used_mb", "mem_free_mb"


# ── hardware helpers ──────────────────────────────────────────────────────────

def _gpu_mean(rows, suffix):
    """Mean across N_GPUS for a given column suffix. Returns None if no data."""
    if suffix is None:
        return None
    vals = []
    for r in rows:
        for i in range(N_GPUS):
            v = r.get(f"gpu{i}_{suffix}")
            if v is not None and isinstance(v, (int, float)):
                vals.append(v)
    return round(statistics.mean(vals), 2) if vals else None


def _col_mean(rows, col):
    vals = [r[col] for r in rows if isinstance(r.get(col), (int, float))]
    return round(statistics.mean(vals), 3) if vals else None


def _col_max(rows, col):
    vals = [r[col] for r in rows if isinstance(r.get(col), (int, float))]
    return round(max(vals), 3) if vals else None


def window_stats(hw, t0, t1, sm_col, bw_col, mem_used_col, mem_free_col):
    """
    Aggregate hardware metrics for samples in [t0, t1].

    Column suffixes (sm_col/bw_col/mem_used_col/mem_free_col) are detected once
    per trace and passed in, so this function handles both hardware.csv formats.

    Returns dict with canonical keys (gpu_sm_util_mean, gpu_mem_bw_mean, …)
    so callers (compute_bubbles, classify_resource) don't need format awareness.
    """
    w = [r for r in hw if t0 <= r["t"] <= t1]
    _none = {
        "gpu_sm_util_mean": None, "gpu_sm_occ_mean": None,
        "gpu_mem_bw_mean": None, "gpu_mem_used_mean_mb": None,
        "gpu_mem_free_mean_mb": None, "gpu_power_mean_w": None,
        "cpu_mean": None, "cpu_max_mean": None, "cpu_hot_cores_mean": None,
        "n_samples": 0,
    }
    if not w:
        return _none

    # Per-GPU row powers (for summed total)
    row_powers = []
    for r in w:
        rp = [r.get(f"gpu{i}_power_w") for i in range(N_GPUS)]
        if all(isinstance(v, (int, float)) for v in rp):
            row_powers.append(sum(rp))

    return {
        "gpu_sm_util_mean":    _gpu_mean(w, sm_col),
        "gpu_sm_occ_mean":     _gpu_mean(w, "sm_occ"),    # DCGM only; None for Gen-1/NVML
        "gpu_mem_bw_mean":     _gpu_mean(w, bw_col),      # None for Gen-1
        "gpu_mem_used_mean_mb": _gpu_mean(w, mem_used_col),
        "gpu_mem_free_mean_mb": _gpu_mean(w, mem_free_col),
        "gpu_power_mean_w":    round(statistics.mean(row_powers), 1) if row_powers else None,
        "cpu_mean":            _col_mean(w, "cpu_mean"),
        "cpu_max_mean":        _col_mean(w, "cpu_max"),
        "cpu_hot_cores_mean":  _col_mean(w, "cpu_hot_cores"),
        "n_samples":           len(w),
    }


def _kv_pct_in_window(sglang_rows, t0, t1):
    """KV cache pool % from sglang_metrics.csv rows in [t0, t1]. None if unavailable."""
    if not sglang_rows:
        return None
    w = [r["token_usage"] * 100.0
         for r in sglang_rows
         if t0 <= r["t"] <= t1 and isinstance(r.get("token_usage"), (int, float))]
    return round(statistics.mean(w), 2) if w else None


# ── span annotation ───────────────────────────────────────────────────────────

def assign_turn_indices(gen_spans, exec_spans):
    """
    turn_idx = -1  → pre-loop generate (before the first execute ever fires)
    turn_idx = k   → generate that follows execute[k] (0-based)
    """
    first_exec_t = exec_spans[0]["t_start"] if exec_spans else float("inf")
    annotated = []
    for g in gen_spans:
        if g["t_end"] <= first_exec_t:
            turn_idx = -1
        else:
            preceding = [i for i, e in enumerate(exec_spans) if e["t_end"] <= g["t_start"]]
            turn_idx = max(preceding) if preceding else -1
        annotated.append({**g, "turn_idx": turn_idx})
    return annotated


def classify_resource(duration_s, hw_s):
    """
    Tag each execute window with its dominant resource type.

    resource_type values:
      trivial      - < 50ms
      gpu_tool     - GPU actively used (embedding, DeepPurpose)
      cpu_tool     - CPU saturated, GPU idle (classic GPU bubble)
      network_tool - both CPU and GPU idle (waiting on network/IO)
      mixed        - partial GPU or ambiguous
      unknown      - no hardware data
    """
    if duration_s < 0.05:
        return "trivial"
    gpu_sm  = hw_s.get("gpu_sm_util_mean")
    gpu_bw  = hw_s.get("gpu_mem_bw_mean")
    cpu_mean = hw_s.get("cpu_mean")
    if gpu_sm is None:
        return "unknown"

    gpu_active = (gpu_sm >= 30) or (gpu_bw is not None and gpu_bw >= 30)
    if gpu_active:
        return "gpu_tool"

    effective_cores = (cpu_mean or 0.0) * N_CPUS / 100.0
    if effective_cores >= 0.5:
        return "cpu_tool"

    if gpu_sm < 10:
        return "network_tool"

    return "mixed"


# ── Layer-3 per-span metrics ──────────────────────────────────────────────────

def per_span_sglang(sglang_rows, t0, t1):
    """
    Compute per-generate-span metrics from sglang_metrics.csv cumulative counters.
    Uses the scrape just before t0 and just after t1 to get deltas.

    Returns {} if sglang_metrics not available or span has no bracketing scrapes.
    Key metrics:
      ttft_s         — time to first token (per request, from engine's view)
      tpot_ms        — per-output-token latency in ms (decode phase)
      queue_time_ms  — time request spent waiting in SGLang queue
                       (~0 single-tenant; grows under multi-tenancy)
      engine_e2e_s   — end-to-end latency from SGLang's perspective
      fwd_extend_s   — GPU time spent on prefill (extend) passes this span
      fwd_decode_s   — GPU time spent on decode passes this span
    """
    if not sglang_rows:
        return {}
    before = [r for r in sglang_rows if r["t"] <= t0]
    after  = [r for r in sglang_rows if r["t"] >= t1]
    if not before or not after:
        return {}
    r0, r1 = before[-1], after[0]

    def delta(key):
        v0, v1 = r0.get(key), r1.get(key)
        if isinstance(v0, (int, float)) and isinstance(v1, (int, float)) and v1 >= v0:
            return v1 - v0
        return None

    def mean_ratio(sum_key, count_key):
        s, c = delta(sum_key), delta(count_key)
        return round(s / c, 5) if s is not None and c and c > 0 else None

    result = {}

    ttft = mean_ratio("ttft_sum", "ttft_count")
    if ttft is not None:
        result["ttft_s"] = ttft

    tpot = mean_ratio("tpot_sum", "tpot_count")
    if tpot is not None:
        result["tpot_ms"] = round(tpot * 1000, 2)

    qt = mean_ratio("queue_time_sum", "queue_time_count")
    if qt is not None:
        result["queue_time_ms"] = round(qt * 1000, 3)

    e2e = mean_ratio("e2e_sum", "e2e_count")
    if e2e is not None:
        result["engine_e2e_s"] = e2e

    # Prefill vs decode GPU time
    ext = delta("fwd_exec_extend_s")
    dec = delta("fwd_exec_decode_s")
    if ext is not None:
        result["fwd_extend_s"] = round(ext, 4)
    if dec is not None:
        result["fwd_decode_s"] = round(dec, 4)

    # Token throughput
    for key in ("tokens_prefill_compute", "tokens_prefill_cache", "tokens_decode"):
        d = delta(key)
        if d is not None:
            result[key] = round(d, 0)

    # Mean queue depth over this span
    w_rows = [r for r in sglang_rows if t0 <= r["t"] <= t1]
    for gauge in ("num_running_reqs", "num_queue_reqs"):
        vals = [r[gauge] for r in w_rows if isinstance(r.get(gauge), (int, float))]
        if vals:
            result[f"{gauge}_mean"] = round(statistics.mean(vals), 2)

    return result


# ── bubble computation ────────────────────────────────────────────────────────

def compute_bubbles(gen_spans, exec_spans, hw, wall_time_s, sm_col, bw_col,
                    mem_used_col, mem_free_col):
    """
    GPU bubble — wasted GPU compute during execute windows.
    CPU bubble — wasted CPU capacity during generate windows.
    Both weighted (fractional idle) and binary (fully idle) variants.
    """
    gpu_weighted = gpu_binary = 0.0
    for ex in exec_spans:
        hw_s = window_stats(hw, ex["t_start"], ex["t_end"],
                            sm_col, bw_col, mem_used_col, mem_free_col)
        dur = ex["duration_s"]
        if dur < 0.001:
            continue
        g = hw_s["gpu_sm_util_mean"] if hw_s["gpu_sm_util_mean"] is not None else 0.0
        gpu_weighted += dur * (1.0 - g / 100.0)
        if g < GPU_SM_BUSY_THRESHOLD:
            gpu_binary += dur

    cpu_weighted = cpu_binary = 0.0
    for gen in gen_spans:
        hw_s = window_stats(hw, gen["t_start"], gen["t_end"],
                            sm_col, bw_col, mem_used_col, mem_free_col)
        dur = gen["duration_s"]
        c = hw_s["cpu_mean"] if hw_s["cpu_mean"] is not None else 0.0
        cpu_weighted += dur * (1.0 - c / 100.0)
        if c < CPU_BUSY_THRESHOLD:
            cpu_binary += dur

    safe = wall_time_s if wall_time_s else 1.0
    return {
        "gpu_weighted_s":        round(gpu_weighted, 3),
        "gpu_weighted_fraction": round(gpu_weighted / safe, 4),
        "gpu_binary_s":          round(gpu_binary, 3),
        "gpu_binary_fraction":   round(gpu_binary / safe, 4),
        "cpu_weighted_s":        round(cpu_weighted, 3),
        "cpu_weighted_fraction": round(cpu_weighted / safe, 4),
        "cpu_binary_s":          round(cpu_binary, 3),
        "cpu_binary_fraction":   round(cpu_binary / safe, 4),
        "thresholds": {
            "gpu_sm_busy_pct": GPU_SM_BUSY_THRESHOLD,
            "cpu_busy_pct":    CPU_BUSY_THRESHOLD,
        },
    }


# ── timeline plot ─────────────────────────────────────────────────────────────

def make_timeline(folder, gen_spans_ann, exec_spans, hw, sglang_rows, meta,
                  sm_col, bw_col):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  matplotlib not installed — skipping timeline.png")
        return

    t_vals    = [r["t"] for r in hw]
    sm_line   = [_gpu_mean([r], sm_col)  for r in hw]
    bw_line   = [_gpu_mean([r], bw_col)  for r in hw] if bw_col else [None] * len(hw)
    cpu_line  = [r.get("cpu_mean")  for r in hw]
    cpu_max_line = [r.get("cpu_max") for r in hw]

    # KV cache from sglang_metrics if available; else try old column in hw rows
    if sglang_rows:
        sg_t   = [r["t"] for r in sglang_rows]
        sg_kv  = [r.get("token_usage") * 100.0
                  if isinstance(r.get("token_usage"), (int, float)) else None
                  for r in sglang_rows]
        kv_t, kv_line = sg_t, sg_kv
    else:
        kv_t = t_vals
        kv_line = [r.get("sglang_kv_cache_pct") for r in hw]

    has_bw = any(v is not None for v in bw_line)
    has_kv = any(v is not None for v in kv_line)

    fig = plt.figure(figsize=(16, 7))
    gs  = gridspec.GridSpec(2, 1, hspace=0.35, height_ratios=[3, 2])
    ax_gpu = fig.add_subplot(gs[0])
    ax_cpu = fig.add_subplot(gs[1])
    ax_kv  = ax_gpu.twinx() if has_kv else None

    def shade_spans(ax):
        for g in gen_spans_ann:
            color = "#aec7e8" if g["turn_idx"] == -1 else "#1f77b4"
            ax.axvspan(g["t_start"], g["t_end"], alpha=0.18, color=color, zorder=1)
        for ex in exec_spans:
            ax.axvspan(ex["t_start"], ex["t_end"], alpha=0.45, color="#ff7f0e", zorder=2)

    shade_spans(ax_gpu)
    shade_spans(ax_cpu)

    sm_label = "GPU SM activity %" if sm_col == "sm_act" else "GPU util %"
    ax_gpu.plot(t_vals, sm_line, color="#2ca02c", lw=1.8, zorder=3, label=sm_label)
    if has_bw:
        ax_gpu.plot(t_vals, bw_line, color="#17becf", lw=1.4, ls="-.", zorder=3,
                    label="HBM DRAM activity % (decode BW bottleneck)")
    if has_kv and ax_kv:
        kv_clean = [v if v is not None else float("nan") for v in kv_line]
        ax_kv.plot(kv_t, kv_clean, color="#9467bd", lw=1.2, ls=":", zorder=3,
                   label="KV cache pool % (SGLang)")
        ax_kv.set_ylim(-5, 105)
        ax_kv.set_ylabel("KV cache pool (%)", color="#9467bd", fontsize=8)
        ax_kv.tick_params(axis="y", labelcolor="#9467bd", labelsize=7)

    ax_gpu.set_ylim(-5, 115)
    ax_gpu.set_ylabel("GPU utilization (%)", fontsize=9)
    ax_gpu.tick_params(axis="x", labelbottom=False)

    ax_cpu.plot(t_vals, cpu_line, color="#d62728", lw=1.2, zorder=3,
                label=f"CPU mean (all {meta.get('n_cpu_sampled', '?')} cores)")
    ax_cpu.plot(t_vals, cpu_max_line, color="#ff9896", lw=1.0, ls="--", zorder=3,
                label="CPU max (hottest core)")
    ax_cpu.set_ylim(-5, 115)
    ax_cpu.set_xlabel("Time (s from trace start)", fontsize=9)
    ax_cpu.set_ylabel("CPU utilization (%)", fontsize=9)

    gpu_legend = [
        mpatches.Patch(color="#1f77b4", alpha=0.45, label="Generate — agent reasoning"),
        mpatches.Patch(color="#aec7e8", alpha=0.45, label="Generate — pre-loop helper"),
        mpatches.Patch(color="#ff7f0e", alpha=0.65, label="Execute — tool call"),
        plt.Line2D([0], [0], color="#2ca02c", lw=1.8, label=sm_label),
    ]
    if has_bw:
        gpu_legend.append(
            plt.Line2D([0], [0], color="#17becf", lw=1.4, ls="-.",
                       label="HBM DRAM activity %"))
    if has_kv:
        gpu_legend.append(
            plt.Line2D([0], [0], color="#9467bd", lw=1.2, ls=":",
                       label="KV cache pool %"))
    ax_gpu.legend(handles=gpu_legend, loc="upper left", fontsize=7, framealpha=0.85)

    cpu_legend = [
        plt.Line2D([0], [0], color="#d62728", lw=1.2, label="CPU mean (all cores)"),
        plt.Line2D([0], [0], color="#ff9896", lw=1.0, ls="--", label="CPU max (hottest core)"),
    ]
    ax_cpu.legend(handles=cpu_legend, loc="upper left", fontsize=7, framealpha=0.85)

    task = (meta.get("task") or "")[:80]
    hw_src = meta.get("hw_source", "?")
    fig.suptitle(f"Trace: {folder.name}  |  hw={hw_src}  |  {task}", fontsize=9)

    out = folder / "timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  timeline  → {out}")


# ── new multi-figure visualization ───────────────────────────────────────────

_RTYPE_COLOR = {
    "cpu_tool":    "#e6550d",
    "gpu_tool":    "#31a354",
    "network_tool":"#3182bd",
    "mixed":       "#756bb1",
    "trivial":     "#bdbdbd",
    "unknown":     "#969696",
}


def _shade_phases(ax, gen_spans_ann, exec_spans):
    for g in gen_spans_ann:
        color = "#aec7e8" if g["turn_idx"] == -1 else "#1f77b4"
        ax.axvspan(g["t_start"], g["t_end"], alpha=0.15, color=color, zorder=1)
    for ex in exec_spans:
        ax.axvspan(ex["t_start"], ex["t_end"], alpha=0.35, color="#ff7f0e", zorder=2)


def _kv_series(hw, sglang_rows):
    if sglang_rows:
        t  = [r["t"] for r in sglang_rows]
        kv = [r["token_usage"] * 100.0
              if isinstance(r.get("token_usage"), (int, float)) else float("nan")
              for r in sglang_rows]
        return t, kv
    t  = [r["t"] for r in hw]
    kv = [r.get("sglang_kv_cache_pct") or float("nan") for r in hw]
    return t, kv


def _fig1_timeline(gen_spans_ann, exec_spans, hw, sglang_rows, meta, sm_col, bw_col):
    """Hardware time-series: GPU SM/DRAM/occupancy + CPU mean/max + KV cache."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    t_vals    = [r["t"] for r in hw]
    sm_line   = [_gpu_mean([r], sm_col)   or float("nan") for r in hw]
    bw_line   = [_gpu_mean([r], bw_col)   or float("nan") for r in hw] if bw_col else None
    occ_line  = [_gpu_mean([r], "sm_occ") or float("nan") for r in hw]
    cpu_line  = [r.get("cpu_mean") or float("nan") for r in hw]
    cpu_max_l = [r.get("cpu_max")  or float("nan") for r in hw]
    kv_t, kv_l = _kv_series(hw, sglang_rows)

    has_bw  = bw_col is not None
    has_occ = any(v == v for v in occ_line)
    has_kv  = any(v == v for v in kv_l)

    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.32, height_ratios=[3, 2])
    ax_gpu = fig.add_subplot(gs[0])
    ax_cpu = fig.add_subplot(gs[1])
    ax_kv  = ax_gpu.twinx() if has_kv else None

    _shade_phases(ax_gpu, gen_spans_ann, exec_spans)
    _shade_phases(ax_cpu, gen_spans_ann, exec_spans)

    sm_label = "SM activity %" if sm_col == "sm_act" else "GPU util %"
    ax_gpu.plot(t_vals, sm_line, color="#2ca02c", lw=1.8, zorder=3, label=sm_label)
    if has_bw:
        ax_gpu.plot(t_vals, bw_line, color="#17becf", lw=1.4, ls="-.", zorder=3,
                    label="DRAM activity % (HBM BW)")
    if has_occ:
        ax_gpu.plot(t_vals, occ_line, color="#8c564b", lw=1.0, ls=":", zorder=3,
                    label="SM occupancy % (low=BW-bound decode)")
    if has_kv and ax_kv:
        ax_kv.plot(kv_t, kv_l, color="#9467bd", lw=1.2, zorder=3)
        ax_kv.set_ylim(-2, 105)
        ax_kv.set_ylabel("KV cache pool %", color="#9467bd", fontsize=8)
        ax_kv.tick_params(axis="y", labelcolor="#9467bd", labelsize=7)

    ax_gpu.set_ylim(-5, 115)
    ax_gpu.set_ylabel("GPU %", fontsize=9)
    ax_gpu.tick_params(axis="x", labelbottom=False)

    ax_cpu.plot(t_vals, cpu_line,  color="#d62728", lw=1.2, zorder=3,
                label=f"CPU mean ({meta.get('n_cpu_sampled','?')} cores)")
    ax_cpu.plot(t_vals, cpu_max_l, color="#fc8d59", lw=1.0, ls="--", zorder=3,
                label="CPU hottest core")
    ax_cpu.set_ylim(-5, 115)
    ax_cpu.set_xlabel("Time (s from trace start)", fontsize=9)
    ax_cpu.set_ylabel("CPU %", fontsize=9)

    hw_src = meta.get("hw_source", "?")
    leg = [
        mpatches.Patch(color="#1f77b4", alpha=0.4,  label="Generate (reasoning)"),
        mpatches.Patch(color="#aec7e8", alpha=0.4,  label="Generate (pre-loop)"),
        mpatches.Patch(color="#ff7f0e", alpha=0.55, label="Execute (tool)"),
        plt.Line2D([0],[0], color="#2ca02c", lw=1.8, label=sm_label),
    ]
    if has_bw:  leg.append(plt.Line2D([0],[0], color="#17becf", lw=1.4, ls="-.", label="DRAM act %"))
    if has_occ: leg.append(plt.Line2D([0],[0], color="#8c564b", lw=1.0, ls=":", label="SM occ %"))
    if has_kv:  leg.append(plt.Line2D([0],[0], color="#9467bd", lw=1.2, label="KV cache %"))
    ax_gpu.legend(handles=leg, loc="upper left", fontsize=7, framealpha=0.85)
    ax_cpu.legend(loc="upper left", fontsize=7, framealpha=0.85)
    fig.suptitle(f"[1] Hardware timeline  |  hw={hw_src}  |  {meta.get('task_id','')}",
                 fontsize=9, fontweight="bold")
    return fig


def _fig2_phase_hardware(gen_hw, exec_hw, sm_col, bw_col, meta):
    """Grouped bar: GPU and CPU metrics during generate vs execute."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels, gen_vals, exe_vals = [], [], []
    def _add(lbl, gv, ev):
        if gv is not None or ev is not None:
            labels.append(lbl); gen_vals.append(gv or 0.0); exe_vals.append(ev or 0.0)

    _add("GPU SM act %",   _gpu_mean(gen_hw, sm_col),    _gpu_mean(exec_hw, sm_col))
    if bw_col:
        _add("GPU DRAM act %", _gpu_mean(gen_hw, bw_col),    _gpu_mean(exec_hw, bw_col))
    _add("GPU SM occ %",   _gpu_mean(gen_hw, "sm_occ"),  _gpu_mean(exec_hw, "sm_occ"))
    _add("CPU mean %",     _col_mean(gen_hw, "cpu_mean"), _col_mean(exec_hw, "cpu_mean"))
    _add("CPU max-core %", _col_mean(gen_hw, "cpu_max"),  _col_mean(exec_hw, "cpu_max"))
    if not labels:
        return None

    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, gen_vals, w, label="Generate", color="#1f77b4", alpha=0.85)
    b2 = ax.bar(x + w/2, exe_vals, w, label="Execute",  color="#ff7f0e", alpha=0.85)
    ax.bar_label(b1, fmt="%.1f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Utilization (%)", fontsize=9); ax.set_ylim(0, 120)
    ax.axhline(GPU_SM_BUSY_THRESHOLD, color="red", lw=0.8, ls="--", alpha=0.5,
               label=f"GPU busy threshold ({GPU_SM_BUSY_THRESHOLD}%)")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.set_title(f"[2] Phase hardware profile  |  {meta.get('task_id','')}\n"
                 "Execute GPU bars near 0 = GPU bubble", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def _fig3_wall_time(analysis, meta):
    """Stacked bar of wall-time breakdown + key metric text panel."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    wt  = analysis["meta"]["wall_time_s"]
    t   = analysis["timing"]
    b   = analysis["bubble"]
    exs = analysis["execute_spans"]

    rtype_times = {}
    for ex in exs:
        rt = ex.get("resource_type", "unknown")
        rtype_times[rt] = rtype_times.get(rt, 0) + ex.get("duration_s", 0)

    segments = [("Init", t["init_time_s"], "#aec7e8"),
                ("Generate", t["generate_total_s"], "#1f77b4")]
    for rt, dur in sorted(rtype_times.items()):
        segments.append((f"Execute ({rt})", dur, _RTYPE_COLOR.get(rt, "#bdbdbd")))
    inter = t.get("inter_phase_s") or 0
    if inter > 0.05:
        segments.append(("Inter-phase", inter, "#e7e7e7"))

    fig, (ax_bar, ax_txt) = plt.subplots(1, 2, figsize=(13, 3.5),
                                          gridspec_kw={"width_ratios": [3, 1]})
    left = 0.0
    for label, dur, color in segments:
        ax_bar.barh(0, dur, left=left, color=color, edgecolor="white", lw=0.5)
        if dur / wt > 0.04:
            ax_bar.text(left + dur/2, 0,
                        f"{label}\n{dur:.1f}s ({100*dur/wt:.0f}%)",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        left += dur
    ax_bar.set_xlim(0, wt); ax_bar.set_yticks([])
    ax_bar.set_xlabel("Time (s)", fontsize=9)
    leg = [mpatches.Patch(color=c, label=l) for l, _, c in segments]
    ax_bar.legend(handles=leg, fontsize=7, loc="lower right",
                  bbox_to_anchor=(1.0, 1.02), ncol=len(segments))
    ax_bar.set_title(f"[3] Wall-time breakdown  |  {meta.get('task_id','')}  |  total={wt:.1f}s",
                     fontsize=9, fontweight="bold")

    l3 = analysis.get("layer3", {})
    lines = [
        f"GPU bubble (binary):    {b['gpu_binary_s']:.1f}s = {100*b['gpu_binary_fraction']:.1f}%",
        f"GPU bubble (weighted):  {b['gpu_weighted_s']:.1f}s = {100*b['gpu_weighted_fraction']:.1f}%",
        f"CPU bubble (weighted):  {b['cpu_weighted_s']:.1f}s = {100*b['cpu_weighted_fraction']:.1f}%",
        "",
        f"Turns (real):   {analysis['turns']['n_real_turns']}",
        f"Generate total: {t['generate_total_s']:.1f}s",
        f"Execute total:  {t['execute_total_s']:.1f}s",
    ]
    if l3.get("available") and l3.get("mean_ttft_s") is not None:
        lines += ["",
                  f"TTFT:        {l3['mean_ttft_s']:.3f}s",
                  f"TPOT:        {l3['mean_tpot_ms']:.1f} ms" if l3.get("mean_tpot_ms") is not None else "TPOT:        n/a",
                  f"Queue wait:  {l3['queue_wait_mean_ms']:.1f} ms" if l3.get("queue_wait_mean_ms") is not None else "Queue wait:  n/a"]
    ax_txt.axis("off")
    ax_txt.text(0.05, 0.97, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=8, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.9))
    fig.tight_layout()
    return fig


def _fig4_per_turn(gen_spans_ann, exec_spans, analysis, meta):
    """Per-turn generate+execute durations (left) and Layer-2 token profile (right)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    real_gens   = [g for g in gen_spans_ann if g["turn_idx"] != -1]
    n           = max(len(real_gens), len(exec_spans))
    if n == 0:
        return None

    turn_ids    = list(range(n))
    gen_durs    = [real_gens[i]["duration_s"] if i < len(real_gens) else 0 for i in turn_ids]
    exec_durs   = [exec_spans[i]["duration_s"] if i < len(exec_spans) else 0 for i in turn_ids]
    exec_rtypes = [analysis["execute_spans"][i].get("resource_type","unknown")
                   if i < len(exec_spans) else "unknown" for i in turn_ids]
    exec_colors = [_RTYPE_COLOR.get(rt,"#bdbdbd") for rt in exec_rtypes]

    l2_calls = analysis.get("layer2", {}).get("per_call", [])
    ncols = 2 if l2_calls else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 4.5), squeeze=False)

    ax_d = axes[0][0]
    x = np.arange(n); w = 0.38
    b1 = ax_d.bar(x - w/2, gen_durs, w, color="#1f77b4", alpha=0.85, label="Generate")
    for xv, dur, col in zip(x + w/2, exec_durs, exec_colors):
        ax_d.bar(xv, dur, w, color=col, alpha=0.85)
    ax_d.bar_label(b1, fmt="%.1fs", fontsize=7, padding=2)
    ax_d.set_xticks(x); ax_d.set_xticklabels([f"Turn {i}" for i in turn_ids], fontsize=9)
    ax_d.set_ylabel("Duration (s)", fontsize=9)
    ax_d.set_title("Generate / Execute duration per turn", fontsize=9, fontweight="bold")
    leg = [mpatches.Patch(color="#1f77b4", label="Generate")]
    seen = set()
    for rt, col in zip(exec_rtypes, exec_colors):
        if rt not in seen:
            leg.append(mpatches.Patch(color=col, label=f"Execute ({rt})")); seen.add(rt)
    ax_d.legend(handles=leg, fontsize=7, framealpha=0.85)

    if l2_calls:
        ax_t = axes[0][1]
        xc     = np.arange(len(l2_calls))
        prompt = [c.get("prompt_tokens") or 0 for c in l2_calls]
        compl  = [c.get("completion_tokens") or 0 for c in l2_calls]
        lats   = [c.get("latency_s") or 0 for c in l2_calls]
        ax_t.bar(xc, prompt, color="#6baed6", alpha=0.85, label="Prompt tokens")
        ax_t.bar(xc, compl, bottom=prompt, color="#fd8d3c", alpha=0.85, label="Completion tokens")
        ax_lat = ax_t.twinx()
        ax_lat.plot(xc, lats, "D--", color="#e31a1c", lw=1.4, ms=6, label="Latency (s)")
        ax_lat.set_ylabel("Latency (s)", color="#e31a1c", fontsize=8)
        ax_lat.tick_params(axis="y", labelcolor="#e31a1c", labelsize=7)
        ax_t.set_xticks(xc)
        ax_t.set_xticklabels([f"call {c['seq']}" for c in l2_calls], fontsize=8)
        ax_t.set_ylabel("Tokens", fontsize=9)
        ax_t.set_title("Layer-2: tokens + latency per LLM call", fontsize=9, fontweight="bold")
        h1, l1 = ax_t.get_legend_handles_labels()
        h2, l2 = ax_lat.get_legend_handles_labels()
        ax_t.legend(h1+h2, l1+l2, fontsize=7, framealpha=0.85)

    fig.suptitle(f"[4] Per-turn profile  |  {meta.get('task_id','')}", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def _fig5_layer3(analysis, sglang_rows, meta):
    """Layer-3 serving latency: TTFT/e2e, TPOT/queue-wait, KV cache + throughput."""
    import matplotlib.pyplot as plt
    import numpy as np

    l3 = analysis.get("layer3", {})
    if not l3.get("available"):
        return None
    p3 = analysis.get("prefill_vs_decode", {}).get("per_generate_span", [])
    if not p3:
        return None

    span_ids = [s["span_idx"] for s in p3]
    ttft  = [s.get("ttft_s") or 0       for s in p3]
    e2e   = [s.get("engine_e2e_s") or 0 for s in p3]
    tpot  = [s.get("tpot_ms") or 0      for s in p3]
    qwait = [s.get("queue_time_ms") or 0 for s in p3]

    n_panels = 3 if sglang_rows else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5*n_panels, 4.5))
    x = np.arange(len(span_ids))

    # Panel A: TTFT and engine e2e
    ax = axes[0]
    ax.bar(x, e2e,  color="#9ecae1", alpha=0.9, label="Engine e2e (s)")
    ax.bar(x, ttft, color="#2171b5", alpha=0.9, label="TTFT (s)")
    ax.set_xticks(x); ax.set_xticklabels([f"gen {i}" for i in span_ids], fontsize=8)
    ax.set_ylabel("Seconds", fontsize=9)
    ax.set_title("TTFT & engine e2e per generate span", fontsize=8, fontweight="bold")
    ax.legend(fontsize=7)
    for i, e in enumerate(e2e):
        ax.text(i, e + 0.05, f"{e:.2f}s", ha="center", fontsize=7)

    # Panel B: TPOT and queue wait
    ax = axes[1]; ax2 = ax.twinx()
    ax.bar(x - 0.2, tpot,  0.35, color="#fd8d3c", alpha=0.9, label="TPOT (ms)")
    ax2.bar(x + 0.2, qwait, 0.35, color="#74c476", alpha=0.9, label="Queue wait (ms)")
    ax2.set_ylabel("Queue wait (ms)", color="#74c476", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#74c476", labelsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"gen {i}" for i in span_ids], fontsize=8)
    ax.set_ylabel("TPOT (ms)", fontsize=9)
    ax.set_title("TPOT & queue wait\n(queue~0 single-tenant)", fontsize=8, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=7)

    # Panel C: KV cache % + decode throughput over time
    if sglang_rows:
        ax = axes[2]
        kv_t, kv_l = _kv_series([], sglang_rows)
        ax.plot(kv_t, kv_l, color="#9467bd", lw=1.4, label="KV cache pool %")
        ax.set_ylabel("KV cache pool %", color="#9467bd", fontsize=8)
        ax.tick_params(axis="y", labelcolor="#9467bd", labelsize=7)
        thr_rows = [(r["t"], r["gen_throughput"]) for r in sglang_rows
                    if isinstance(r.get("gen_throughput"), (int, float))]
        if thr_rows:
            thr_t, thr_v = zip(*thr_rows)
            ax3 = ax.twinx()
            ax3.plot(thr_t, thr_v, color="#e6550d", lw=1.0, ls="--",
                     alpha=0.8, label="Gen throughput (tok/s)")
            ax3.set_ylabel("Tokens/s", color="#e6550d", fontsize=8)
            ax3.tick_params(axis="y", labelcolor="#e6550d", labelsize=7)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title("KV cache % & decode throughput over time", fontsize=8, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"[5] Layer-3 SGLang engine  |  {meta.get('task_id','')}\n"
        f"mean TTFT={l3.get('mean_ttft_s','?')}s  "
        f"TPOT={l3.get('mean_tpot_ms','?')}ms  "
        f"queue_wait={l3.get('queue_wait_mean_ms','?')}ms",
        fontsize=9, fontweight="bold"
    )
    fig.tight_layout()
    return fig


def _fig6_dcgm(hw, gen_spans_ann, exec_spans, meta):
    """DCGM-only deep-dive: total power, per-GPU SM occupancy, HBM free."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    has_power = any(isinstance(r.get("gpu0_power_w"), (int, float)) for r in hw)
    has_occ   = any(isinstance(r.get("gpu0_sm_occ"),  (int, float)) for r in hw)
    has_mem   = any(isinstance(r.get("gpu0_mem_free_mb"), (int, float)) for r in hw)
    n = sum([has_power, has_occ, has_mem])
    if n == 0:
        return None

    t_vals = [r["t"] for r in hw]
    fig = plt.figure(figsize=(16, 3.5 * n))
    gs  = gridspec.GridSpec(n, 1, hspace=0.45)
    panel = 0

    if has_power:
        ax = fig.add_subplot(gs[panel]); panel += 1
        _shade_phases(ax, gen_spans_ann, exec_spans)
        total_pw = [
            sum(r.get(f"gpu{i}_power_w") or 0 for i in range(N_GPUS))
            if all(isinstance(r.get(f"gpu{i}_power_w"), (int, float)) for i in range(N_GPUS))
            else float("nan")
            for r in hw
        ]
        ax.plot(t_vals, total_pw, color="#d62728", lw=1.4)
        ax.set_ylabel("Total GPU power (W)", fontsize=8)
        ax.set_title("Total GPU power (4x A100): generate=high, execute=idle", fontsize=8)
        if panel < n: ax.tick_params(axis="x", labelbottom=False)

    if has_occ:
        ax = fig.add_subplot(gs[panel]); panel += 1
        _shade_phases(ax, gen_spans_ann, exec_spans)
        for i in range(N_GPUS):
            vals = [r.get(f"gpu{i}_sm_occ") or float("nan") for r in hw]
            ax.plot(t_vals, vals, lw=1.0, alpha=0.75, label=f"GPU{i}")
        ax.set_ylabel("SM occupancy %", fontsize=8)
        ax.set_title("SM occupancy per GPU (low during decode = BW-bound, not compute-bound)", fontsize=8)
        ax.legend(fontsize=6, loc="upper right", ncol=4)
        if panel < n: ax.tick_params(axis="x", labelbottom=False)

    if has_mem:
        ax = fig.add_subplot(gs[panel]); panel += 1
        _shade_phases(ax, gen_spans_ann, exec_spans)
        free_mean = [_gpu_mean([r], "mem_free_mb") or float("nan") for r in hw]
        ax.plot(t_vals, free_mean, color="#2171b5", lw=1.4)
        ax.set_ylabel("HBM free (MB, mean 4 GPUs)", fontsize=8)
        ax.set_title("HBM headroom: remaining capacity for a co-scheduled tenant's KV cache", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(f"[6] DCGM deep-dive  |  {meta.get('task_id','')}", fontsize=9, fontweight="bold")
    return fig


def make_figures(folder, gen_spans_ann, exec_spans, hw, sglang_rows,
                 meta, sm_col, bw_col, analysis):
    """
    Generate 6 individual figures + one overview tiling all of them.
    Skips figures that have no data (returns None from their function).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except ImportError:
        print("  matplotlib not installed — skipping figures")
        return

    gen_hw  = [r for r in hw if any(g["t_start"] <= r["t"] <= g["t_end"] for g in gen_spans_ann)]
    exec_hw = [r for r in hw if any(e["t_start"] <= r["t"] <= e["t_end"] for e in exec_spans)]

    specs = [
        ("fig_1_timeline.png",     _fig1_timeline(gen_spans_ann, exec_spans, hw, sglang_rows, meta, sm_col, bw_col)),
        ("fig_2_phase_hardware.png", _fig2_phase_hardware(gen_hw, exec_hw, sm_col, bw_col, meta)),
        ("fig_3_wall_time.png",    _fig3_wall_time(analysis, meta)),
        ("fig_4_per_turn.png",     _fig4_per_turn(gen_spans_ann, exec_spans, analysis, meta)),
        ("fig_5_layer3.png",       _fig5_layer3(analysis, sglang_rows, meta)),
        ("fig_6_dcgm.png",         _fig6_dcgm(hw, gen_spans_ann, exec_spans, meta)),
    ]

    saved = []
    for fname, fig in specs:
        if fig is None:
            continue
        out = folder / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
        print(f"  figure    -> {out.name}")

    # Overview: tile all saved PNGs into one image
    if len(saved) < 2:
        return
    cols = 2
    rows = (len(saved) + 1) // 2
    fig_ov, axes = plt.subplots(rows, cols, figsize=(cols * 14, rows * 8))
    for ax, path in zip(axes.flat, saved):
        ax.imshow(mpimg.imread(str(path)))
        ax.axis("off")
        ax.set_title(path.stem.replace("_", " "), fontsize=8, pad=2)
    for ax in list(axes.flat)[len(saved):]:
        ax.set_visible(False)
    task_short = (meta.get("task") or "")[:90]
    fig_ov.suptitle(
        f"Biomni profiler overview  |  {folder.name}\n{task_short}",
        fontsize=11, fontweight="bold", y=1.001
    )
    fig_ov.tight_layout(pad=0.5)
    out_ov = folder / "fig_overview.png"
    fig_ov.savefig(out_ov, dpi=120, bbox_inches="tight")
    plt.close(fig_ov)
    print(f"  overview  -> {out_ov.name}")


# ── main analysis ─────────────────────────────────────────────────────────────

def analyze(folder: Path, plot: bool = True):
    meta    = json.loads((folder / "meta.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())

    gen_spans, exec_spans, layer2_calls = load_events(folder)
    hw           = load_hw(folder)
    sglang_rows  = load_sglang_metrics(folder)
    wall_time_s  = summary["wall_time_s"]

    # ── Detect hardware column format ─────────────────────────────────────────
    sm_col, bw_col, mem_used_col, mem_free_col = _detect_hw_format(hw)
    hw_gen = "gen2" if sm_col == "sm_act" else "gen1"
    print(f"  hw format  {hw_gen}  (sm_col={sm_col}, bw_col={bw_col})")
    print(f"  sglang_metrics  {len(sglang_rows)} rows  {'(Layer-3 available)' if sglang_rows else '(not present)'}")
    print(f"  layer2_calls    {len(layer2_calls)}")

    gen_spans_ann = assign_turn_indices(gen_spans, exec_spans)
    n_pre_loop   = sum(1 for g in gen_spans_ann if g["turn_idx"] == -1)
    n_real_turns = len(exec_spans)

    # ── Per-span hardware + Layer-3 stats ─────────────────────────────────────
    gen_detail = []
    for i, g in enumerate(gen_spans_ann):
        hw_s = window_stats(hw, g["t_start"], g["t_end"],
                            sm_col, bw_col, mem_used_col, mem_free_col)
        l3 = per_span_sglang(sglang_rows, g["t_start"], g["t_end"])
        kv = _kv_pct_in_window(sglang_rows, g["t_start"], g["t_end"])
        rec = {
            "idx":              i,
            "turn_idx":         g["turn_idx"],
            "t_start":          round(g["t_start"], 4),
            "t_end":            round(g["t_end"],   4),
            "duration_s":       g["duration_s"],
            "gpu_sm_util":      hw_s["gpu_sm_util_mean"],
            "gpu_sm_occ":       hw_s["gpu_sm_occ_mean"],
            "gpu_mem_bw_util":  hw_s["gpu_mem_bw_mean"],
            "gpu_mem_used_mb":  hw_s["gpu_mem_used_mean_mb"],
            "gpu_mem_free_mb":  hw_s["gpu_mem_free_mean_mb"],
            "gpu_power_w":      hw_s["gpu_power_mean_w"],
            "kv_cache_pct":     kv,
            "cpu_mean":         hw_s["cpu_mean"],
            "cpu_max":          hw_s["cpu_max_mean"],
            "cpu_hot_cores":    hw_s["cpu_hot_cores_mean"],
            "n_samples":        hw_s["n_samples"],
        }
        if l3:
            rec.update(l3)
        gen_detail.append(rec)

    exec_detail = []
    for i, ex in enumerate(exec_spans):
        hw_s  = window_stats(hw, ex["t_start"], ex["t_end"],
                             sm_col, bw_col, mem_used_col, mem_free_col)
        kv    = _kv_pct_in_window(sglang_rows, ex["t_start"], ex["t_end"])
        rtype = classify_resource(ex["duration_s"], hw_s)
        gpu_sm = hw_s["gpu_sm_util_mean"] or 0.0
        gpu_idle_s = ex["duration_s"] * (1.0 - gpu_sm / 100.0)
        exec_detail.append({
            "idx":              i,
            "turn_idx":         i,
            "t_start":          round(ex["t_start"], 4),
            "t_end":            round(ex["t_end"],   4),
            "duration_s":       ex["duration_s"],
            "fn":               ex.get("fn"),
            "lang":             ex.get("lang"),
            "resource_type":    rtype,
            "gpu_idle_s":       round(gpu_idle_s, 4),
            "gpu_sm_util":      hw_s["gpu_sm_util_mean"],
            "gpu_sm_occ":       hw_s["gpu_sm_occ_mean"],
            "gpu_mem_bw_util":  hw_s["gpu_mem_bw_mean"],
            "gpu_mem_used_mb":  hw_s["gpu_mem_used_mean_mb"],
            "gpu_mem_free_mb":  hw_s["gpu_mem_free_mean_mb"],
            "gpu_power_w":      hw_s["gpu_power_mean_w"],
            "kv_cache_pct":     kv,
            "cpu_mean":         hw_s["cpu_mean"],
            "cpu_max":          hw_s["cpu_max_mean"],
            "cpu_hot_cores":    hw_s["cpu_hot_cores_mean"],
            "n_samples":        hw_s["n_samples"],
        })

    # ── Phase-aggregate hardware rows ─────────────────────────────────────────
    gen_hw  = [r for r in hw if any(g["t_start"] <= r["t"] <= g["t_end"] for g in gen_spans)]
    exec_hw = [r for r in hw if any(e["t_start"] <= r["t"] <= e["t_end"] for e in exec_spans)]

    gen_durs  = [g["duration_s"] for g in gen_spans]
    exec_durs = [e["duration_s"] for e in exec_spans if e["duration_s"] > 0.001]

    first_event_t = gen_spans[0]["t_start"] if gen_spans else 0.0
    last_event_t  = max(
        gen_spans[-1]["t_end"]  if gen_spans  else 0.0,
        exec_spans[-1]["t_end"] if exec_spans else 0.0,
    )
    agent_active_s = last_event_t - first_event_t

    bubbles = compute_bubbles(gen_spans, exec_spans, hw, wall_time_s,
                              sm_col, bw_col, mem_used_col, mem_free_col)

    # ── KV cache trajectory ───────────────────────────────────────────────────
    if sglang_rows:
        kv_vals = [r["token_usage"] * 100.0
                   for r in sglang_rows
                   if isinstance(r.get("token_usage"), (int, float))]
    else:
        kv_vals = [r["sglang_kv_cache_pct"] for r in hw
                   if isinstance(r.get("sglang_kv_cache_pct"), (int, float))]
    kv_peak = round(max(kv_vals), 2) if kv_vals else None
    kv_end  = round(kv_vals[-1], 2)  if kv_vals else None

    # ── Layer-2 aggregate ─────────────────────────────────────────────────────
    l2_latencies    = [c["latency_s"] for c in layer2_calls if c.get("latency_s") is not None]
    l2_prompt       = [c["prompt_tokens"] for c in layer2_calls if c.get("prompt_tokens") is not None]
    l2_completion   = [c["completion_tokens"] for c in layer2_calls if c.get("completion_tokens") is not None]

    layer2 = {
        "available":               bool(layer2_calls),
        "n_llm_calls":             len(layer2_calls),
        "total_prompt_tokens":     int(sum(l2_prompt))      if l2_prompt     else None,
        "total_completion_tokens": int(sum(l2_completion))  if l2_completion else None,
        "mean_latency_s":          round(statistics.mean(l2_latencies),  3) if l2_latencies  else None,
        "mean_prompt_tokens":      round(statistics.mean(l2_prompt),     1) if l2_prompt     else None,
        "mean_completion_tokens":  round(statistics.mean(l2_completion), 1) if l2_completion else None,
        "per_call":                layer2_calls,
    }

    # ── Layer-3 aggregate ─────────────────────────────────────────────────────
    # Compute per-span Layer-3 metrics and aggregate across real generate spans
    # (exclude pre-loop generates which are Biomni helper calls, not reasoning).
    real_gen_spans = [g for g in gen_spans_ann if g["turn_idx"] != -1]
    l3_spans = [per_span_sglang(sglang_rows, g["t_start"], g["t_end"])
                for g in real_gen_spans]

    def _l3_mean(key):
        vals = [s[key] for s in l3_spans if isinstance(s.get(key), (int, float))]
        return round(statistics.mean(vals), 4) if vals else None

    # Queue-wait: Layer-2 end-to-end minus SGLang engine e2e (single-tenant ≈ 0)
    # Computed per-call where both are available.
    queue_wait_vals = []
    for g, l3 in zip(real_gen_spans, l3_spans):
        if not l3:
            continue
        # Match by time: find the layer2_call with seq closest to this generate span
        # (we can't perfectly match seq to gen_span without seq on events.jsonl start)
        l2_e2e = g["duration_s"]   # Layer-2 latency ≈ span duration
        l3_e2e = l3.get("engine_e2e_s")
        if l3_e2e is not None:
            queue_wait_vals.append(max(0.0, l2_e2e - l3_e2e) * 1000.0)

    layer3 = {
        "available":           bool(sglang_rows),
        "n_scrape_rows":       len(sglang_rows),
        "kv_peak_pct":         kv_peak,
        "kv_end_pct":          kv_end,
        "mean_ttft_s":         _l3_mean("ttft_s"),
        "mean_tpot_ms":        _l3_mean("tpot_ms"),
        "mean_queue_time_ms":  _l3_mean("queue_time_ms"),
        "mean_engine_e2e_s":   _l3_mean("engine_e2e_s"),
        # Queue-wait = Continuum's scheduling bubble; ~0 single-tenant
        "queue_wait_mean_ms":  round(statistics.mean(queue_wait_vals), 2) if queue_wait_vals else None,
        "queue_wait_note":     (
            "queue_wait = Layer-2 span duration − SGLang engine e2e latency. "
            "Should be ≈0 single-tenant; grows under multi-tenancy (Continuum's bubble)."
        ),
    }

    # Prefill vs decode: aggregate fwd_extend and fwd_decode across all real generate spans
    total_extend = sum(s["fwd_extend_s"] for s in l3_spans if isinstance(s.get("fwd_extend_s"), (int, float)))
    total_decode = sum(s["fwd_decode_s"] for s in l3_spans if isinstance(s.get("fwd_decode_s"), (int, float)))
    l3_gen_hw = [r for r in hw
                 if any(g["t_start"] <= r["t"] <= g["t_end"] for g in real_gen_spans)]
    prefill_vs_decode = {
        "available":          bool(sglang_rows),
        "total_fwd_extend_s": round(total_extend, 4) if total_extend else None,
        "total_fwd_decode_s": round(total_decode, 4) if total_decode else None,
        "extend_fraction":    (round(total_extend / (total_extend + total_decode), 4)
                               if total_extend and total_decode else None),
        # DRAM activity during generate spans: high → decode BW-bound (the expected six-axis finding)
        "dram_act_during_generate_mean": _gpu_mean(l3_gen_hw, bw_col) if bw_col else None,
        "sm_occ_during_generate_mean":   _gpu_mean(l3_gen_hw, "sm_occ"),
        "interpretation": (
            "During prefill (extend): SM occupancy high, DRAM activity moderate. "
            "During decode: SM occupancy low, DRAM activity near 100% — BW-bound. "
            "This confirms the six-axis model: GPU bubble is HBM-bandwidth, not compute."
        ),
        "per_generate_span": [
            {"span_idx": i, **l3}
            for i, l3 in enumerate(l3_spans) if l3
        ],
    }

    # ── GPU saturation section ────────────────────────────────────────────────
    gpu_saturation = {
        "during_generate": {
            "sm_util_mean":       _gpu_mean(gen_hw, sm_col),
            "sm_occ_mean":        _gpu_mean(gen_hw, "sm_occ"),
            "mem_bw_util_mean":   _gpu_mean(gen_hw, bw_col),
            "mem_used_mean_mb":   _gpu_mean(gen_hw, mem_used_col),
            "mem_free_mean_mb":   _gpu_mean(gen_hw, mem_free_col),
            "power_mean_w": (
                _col_mean(
                    [{"total_power": sum(r.get(f"gpu{i}_power_w") or 0 for i in range(N_GPUS))}
                     for r in gen_hw],
                    "total_power"
                ) if gen_hw else None
            ),
            "interpretation": (
                "High SM util + high DRAM activity = memory-BW-bound (decode). "
                "sm_occ distinguishes prefill (high) from decode (low, BW-limited). "
                "mem_free_mb = HBM headroom for a co-scheduled tenant's KV cache."
            ),
        },
        "during_execute": {
            "sm_util_mean":       _gpu_mean(exec_hw, sm_col),
            "sm_occ_mean":        _gpu_mean(exec_hw, "sm_occ"),
            "mem_bw_util_mean":   _gpu_mean(exec_hw, bw_col),
            "mem_used_mean_mb":   _gpu_mean(exec_hw, mem_used_col),
            "mem_free_mean_mb":   _gpu_mean(exec_hw, mem_free_col),
            "interpretation": (
                "Low SM util + low DRAM activity = GPU bubble. "
                "This is the window a co-scheduler fills with another tenant's inference."
            ),
        },
        "kv_cache": {
            "peak_pct":  kv_peak,
            "end_pct":   kv_end,
            "source":    "sglang_metrics" if sglang_rows else "hardware.csv",
            "note": (
                "KV cache pool % at peak and end of task. "
                "Remaining capacity shows how much context a concurrent tenant could hold."
            ),
        },
        "hw_format": hw_gen,
    }

    # ── CPU utilization section ───────────────────────────────────────────────
    cpu_utilization = {
        "during_generate": {
            "mean":       _col_mean(gen_hw, "cpu_mean"),
            "max_core":   _col_mean(gen_hw, "cpu_max"),
            "hot_cores":  _col_mean(gen_hw, "cpu_hot_cores"),
        },
        "during_execute": {
            "mean":       _col_mean(exec_hw, "cpu_mean"),
            "max_core":   _col_mean(exec_hw, "cpu_max"),
            "hot_cores":  _col_mean(exec_hw, "cpu_hot_cores"),
        },
        "peak_cpu_mean": _col_max(hw, "cpu_mean"),
        "peak_cpu_max":  _col_max(hw, "cpu_max"),
    }

    # ── Assemble analysis dict ────────────────────────────────────────────────
    analysis = {
        "trace_id": folder.name,
        "meta": {
            "task":                      meta.get("task"),
            "task_id":                   meta.get("task_id"),
            "task_family":               meta.get("task_family"),
            "resource_profile_expected": meta.get("resource_profile_expected"),
            "model":                     meta.get("model"),
            "date":                      (meta.get("t0_wall_iso") or "")[:10],
            "t0_wall_iso":               meta.get("t0_wall_iso"),
            "wall_time_s":               wall_time_s,
            "hw_interval_s":             meta.get("hw_interval_s"),
            "hw_source":                 meta.get("hw_source"),
            "hw_format":                 hw_gen,
            "n_gpu_sampled":             meta.get("n_gpu_sampled"),
            "n_cpu_sampled":             meta.get("n_cpu_sampled"),
            "n_hw_samples":              len(hw),
            "n_sglang_scrapes":          len(sglang_rows),
            "error":                     summary.get("error"),
        },
        "turns": {
            "n_generate_spans":     len(gen_spans),
            "n_execute_spans":      len(exec_spans),
            "n_pre_loop_generates": n_pre_loop,
            "n_real_turns":         n_real_turns,
        },
        "timing": {
            "init_time_s":      round(first_event_t, 3),
            "agent_active_s":   round(agent_active_s, 3),
            "generate_total_s": round(sum(gen_durs), 3),
            "execute_total_s":  round(sum(e["duration_s"] for e in exec_spans), 3),
            "inter_phase_s":    round(
                agent_active_s - sum(gen_durs) - sum(e["duration_s"] for e in exec_spans), 3
            ),
            "generate_mean_s":  round(statistics.mean(gen_durs), 3)   if gen_durs          else None,
            "generate_std_s":   round(statistics.stdev(gen_durs), 3)  if len(gen_durs) > 1 else None,
            "execute_mean_s":   round(statistics.mean(exec_durs), 3)  if exec_durs         else None,
            "execute_std_s":    round(statistics.stdev(exec_durs), 3) if len(exec_durs) > 1 else None,
        },
        "bubble": {
            **bubbles,
            "layer1_execute_total_s":     summary.get("execute_time_s"),
            "layer1_gpu_bubble_fraction": round(
                summary.get("gpu_bubble_time_s_layer1_estimate", 0) / wall_time_s, 4
            ) if wall_time_s else None,
        },
        "layer2":            layer2,
        "layer3":            layer3,
        "prefill_vs_decode": prefill_vs_decode,
        "gpu_saturation":    gpu_saturation,
        "cpu_utilization":   cpu_utilization,
        "generate_spans":    gen_detail,
        "execute_spans":     exec_detail,
    }

    out = folder / "analysis.json"
    out.write_text(json.dumps(analysis, indent=2))
    print(f"  analysis  → {out}")

    if plot:
        make_figures(folder, gen_spans_ann, exec_spans, hw, sglang_rows,
                     meta, sm_col, bw_col, analysis)

    # ── Terminal summary ──────────────────────────────────────────────────────
    t  = analysis["timing"]
    b  = analysis["bubble"]
    gs = analysis["gpu_saturation"]
    c  = analysis["cpu_utilization"]
    l2 = analysis["layer2"]
    l3 = analysis["layer3"]
    wt = wall_time_s

    g_gen  = gs["during_generate"]
    g_exec = gs["during_execute"]
    c_gen  = c["during_generate"]
    c_exec = c["during_execute"]

    print()
    print("═" * 68)
    print(f"  {folder.name}")
    print(f"  {(meta.get('task') or '')[:64]}")
    print(f"  hw={hw_gen}  src={meta.get('hw_source','?')}")
    print("─" * 68)
    print(f"  Wall time       {wt:.1f}s")
    print(f"  Turns           {n_real_turns} real  |  {n_pre_loop} pre-loop generates")
    print(f"  Init overhead   {t['init_time_s']:.2f}s")
    print()
    print(f"  Generate total  {t['generate_total_s']:.2f}s  ({100*t['generate_total_s']/wt:.1f}%)")
    print(f"  Execute total   {t['execute_total_s']:.3f}s  ({100*t['execute_total_s']/wt:.1f}%)")
    print(f"  Inter-phase     {t['inter_phase_s']:.3f}s")
    print()
    sm_lbl = "SM act" if sm_col == "sm_act" else "util  "
    print(f"  GPU during GENERATE (LLM inference):")
    print(f"    {sm_lbl}      {g_gen['sm_util_mean'] or '?':>6}%")
    if g_gen.get('sm_occ_mean') is not None:
        print(f"    SM occ      {g_gen['sm_occ_mean']:>6}%  (high=compute-bound prefill; low=BW-bound decode)")
    if g_gen.get('mem_bw_util_mean') is not None:
        print(f"    DRAM act    {g_gen['mem_bw_util_mean']:>6}%  (real decode bottleneck)")
    print(f"    HBM used    {g_gen['mem_used_mean_mb'] or '?':>6} MB")
    print(f"    HBM free    {g_gen['mem_free_mean_mb'] or '?':>6} MB  (co-scheduling headroom)")
    if gs["kv_cache"]["peak_pct"] is not None:
        print(f"    KV cache    peak {gs['kv_cache']['peak_pct']}%  /  end {gs['kv_cache']['end_pct']}%")
    print()
    print(f"  GPU during EXECUTE (tool calls):")
    print(f"    {sm_lbl}      {g_exec['sm_util_mean'] or '?':>6}%  (GPU bubble — should be near 0)")
    if g_exec.get('mem_bw_util_mean') is not None:
        print(f"    DRAM act    {g_exec['mem_bw_util_mean']:>6}%")
    print()
    print(f"  CPU during GENERATE:")
    print(f"    mean {c_gen['mean'] or '?':>5}%   max-core {c_gen['max_core'] or '?':>5}%   hot-cores {c_gen['hot_cores'] or '?'}")
    print(f"  CPU during EXECUTE:")
    print(f"    mean {c_exec['mean'] or '?':>5}%   max-core {c_exec['max_core'] or '?':>5}%   hot-cores {c_exec['hot_cores'] or '?'}")
    print()
    print(f"  GPU bubble (weighted)  {b['gpu_weighted_s']:.3f}s  ({100*b['gpu_weighted_fraction']:.1f}% of wall)")
    print(f"  GPU bubble (binary)    {b['gpu_binary_s']:.3f}s  ({100*b['gpu_binary_fraction']:.1f}% of wall)")
    print(f"  CPU bubble (weighted)  {b['cpu_weighted_s']:.2f}s  ({100*b['cpu_weighted_fraction']:.1f}% of wall)")
    print()
    if l2["available"]:
        print(f"  Layer-2 (HTTP tokens):")
        print(f"    {l2['n_llm_calls']} LLM calls  "
              f"prompt={l2['total_prompt_tokens']}  "
              f"completion={l2['total_completion_tokens']}  "
              f"mean-latency={l2['mean_latency_s']}s")
    if l3["available"]:
        print(f"  Layer-3 (SGLang):")
        print(f"    TTFT={l3['mean_ttft_s']}s  "
              f"TPOT={l3['mean_tpot_ms']}ms  "
              f"queue_wait={l3['queue_wait_mean_ms']}ms")
        if prefill_vs_decode["total_fwd_extend_s"] is not None:
            print(f"    fwd_extend={prefill_vs_decode['total_fwd_extend_s']}s  "
                  f"fwd_decode={prefill_vs_decode['total_fwd_decode_s']}s  "
                  f"extend_frac={prefill_vs_decode['extend_fraction']}")
    print()
    types = [e["resource_type"] for e in exec_detail]
    print(f"  Execute types  {types}")
    print("═" * 68)

    return analysis


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze one trace run folder.")
    p.add_argument("folder",    help="path to trace run folder")
    p.add_argument("--no-plot", action="store_true", help="skip timeline.png generation")
    args = p.parse_args()
    analyze(Path(args.folder), plot=not args.no_plot)
