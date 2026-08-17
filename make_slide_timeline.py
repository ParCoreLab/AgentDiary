#!/usr/bin/env python3
"""
Generate a presentation-quality timeline figure for random_forest_cv_large.

Output: rf_timeline_slide.png  (in ~/biomni-profiling/)

Legend is placed outside the plot area, all text is large enough to read
from a distance on a slide.
"""

import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── config ────────────────────────────────────────────────────────────────────

TRACE_DIR = Path("results/scrna_pbmc3k/2026-06-27_16-33-13")
OUT_PATH  = Path("scrna_timeline_slide.png")
N_GPUS    = 4

# Font sizes for slide readability
FS_TITLE  = 22
FS_AXIS   = 18
FS_TICK   = 15
FS_LEGEND = 17

# ── data loading ──────────────────────────────────────────────────────────────

def load_events(folder):
    raw = [json.loads(l) for l in (folder / "events.jsonl").read_text().splitlines() if l.strip()]
    pending = {}
    gen_spans, exec_spans = [], []
    for e in raw:
        if e["edge"] == "start":
            pending[e["phase"]] = e
        elif e["edge"] == "end" and e["phase"] in pending:
            s = pending.pop(e["phase"])
            span = {"t_start": s["t"], "t_end": e["t"], "duration_s": e["t"] - s["t"],
                    "fn": s.get("fn") or e.get("fn")}
            if e["phase"] == "generate":
                gen_spans.append(span)
            else:
                exec_spans.append(span)
    return (sorted(gen_spans, key=lambda x: x["t_start"]),
            sorted(exec_spans, key=lambda x: x["t_start"]))


def load_hw(folder):
    rows = []
    with open(folder / "hardware.csv") as f:
        for r in csv.DictReader(f):
            parsed = {}
            for k, v in r.items():
                try:
                    parsed[k] = float(v) if v != "" else None
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def load_sglang(folder):
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


def load_meta(folder):
    p = folder / "meta.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def gpu_mean(rows, suffix):
    if suffix is None:
        return None
    vals = []
    for r in rows:
        for i in range(N_GPUS):
            v = r.get(f"gpu{i}_{suffix}")
            if isinstance(v, (int, float)):
                vals.append(v)
    return statistics.mean(vals) if vals else None


# ── turn annotation ───────────────────────────────────────────────────────────

def annotate_turns(gen_spans, exec_spans):
    first_exec_t = exec_spans[0]["t_start"] if exec_spans else float("inf")
    result = []
    for g in gen_spans:
        if g["t_end"] <= first_exec_t:
            turn_idx = -1
        else:
            preceding = [i for i, e in enumerate(exec_spans) if e["t_end"] <= g["t_start"]]
            turn_idx = max(preceding) if preceding else -1
        result.append({**g, "turn_idx": turn_idx})
    return result


# ── KV series ─────────────────────────────────────────────────────────────────

def kv_series(hw, sglang_rows):
    if sglang_rows:
        t  = [r["t"] for r in sglang_rows]
        kv = [r["token_usage"] * 100.0
              if isinstance(r.get("token_usage"), (int, float)) else float("nan")
              for r in sglang_rows]
        return t, kv
    t  = [r["t"] for r in hw]
    kv = [r.get("sglang_kv_cache_pct") or float("nan") for r in hw]
    return t, kv


# ── shading ───────────────────────────────────────────────────────────────────

def shade_phases(ax, gen_spans_ann, exec_spans):
    for g in gen_spans_ann:
        color = "#aec7e8" if g["turn_idx"] == -1 else "#1f77b4"
        ax.axvspan(g["t_start"], g["t_end"], alpha=0.18, color=color, zorder=1)
    for ex in exec_spans:
        ax.axvspan(ex["t_start"], ex["t_end"], alpha=0.38, color="#ff7f0e", zorder=2)


# ── main figure ───────────────────────────────────────────────────────────────

def make_slide_figure(folder, out_path):
    gen_spans, exec_spans = load_events(folder)
    hw         = load_hw(folder)
    sglang     = load_sglang(folder)
    meta       = load_meta(folder)
    gen_ann    = annotate_turns(gen_spans, exec_spans)

    # detect Gen-2 vs Gen-1
    if hw and "gpu0_sm_act" in hw[0]:
        sm_col, bw_col = "sm_act", "dram_act"
        sm_label = "SM activity % (GPU compute)"
    else:
        sm_col, bw_col = "util", None
        sm_label = "GPU utilization %"

    t_vals   = [r["t"] for r in hw]
    sm_line  = [gpu_mean([r], sm_col) or float("nan") for r in hw]
    bw_line  = [gpu_mean([r], bw_col) or float("nan") for r in hw] if bw_col else None
    occ_line = [gpu_mean([r], "sm_occ") or float("nan") for r in hw]
    cpu_line = [r.get("cpu_mean") or float("nan") for r in hw]
    cpu_max  = [r.get("cpu_max")  or float("nan") for r in hw]
    kv_t, kv_l = kv_series(hw, sglang)

    has_bw  = bw_col is not None
    has_occ = any(v == v for v in occ_line)  # any non-nan
    has_kv  = any(v == v for v in kv_l)

    # ── layout: wide figure with room for external legend on right ─────────────
    fig = plt.figure(figsize=(22, 9))
    # Two rows of plots, 75% width; legend column takes remaining 25%
    gs = gridspec.GridSpec(2, 2, hspace=0.35, width_ratios=[3, 1], height_ratios=[3, 2])
    ax_gpu = fig.add_subplot(gs[0, 0])
    ax_cpu = fig.add_subplot(gs[1, 0])
    ax_leg_top = fig.add_subplot(gs[0, 1])
    ax_leg_bot = fig.add_subplot(gs[1, 1])
    for ax in (ax_leg_top, ax_leg_bot):
        ax.axis("off")

    ax_kv = ax_gpu.twinx() if has_kv else None

    # ── shade phases ───────────────────────────────────────────────────────────
    shade_phases(ax_gpu, gen_ann, exec_spans)
    shade_phases(ax_cpu, gen_ann, exec_spans)

    # ── GPU panel ──────────────────────────────────────────────────────────────
    ax_gpu.plot(t_vals, sm_line,  color="#2ca02c", lw=2.5, zorder=3)
    if has_bw:
        ax_gpu.plot(t_vals, bw_line,  color="#17becf", lw=2.0, ls="-.", zorder=3)
    if has_occ:
        ax_gpu.plot(t_vals, occ_line, color="#8c564b", lw=1.6, ls=":",  zorder=3)
    if has_kv and ax_kv:
        ax_kv.plot(kv_t, kv_l, color="#9467bd", lw=1.8, zorder=3)
        ax_kv.set_ylim(-2, 105)
        ax_kv.set_ylabel("KV cache pool %", color="#9467bd", fontsize=FS_AXIS - 2)
        ax_kv.tick_params(axis="y", labelcolor="#9467bd", labelsize=FS_TICK - 2)

    ax_gpu.set_ylim(-5, 115)
    ax_gpu.set_ylabel("GPU %", fontsize=FS_AXIS)
    ax_gpu.tick_params(axis="x", labelbottom=False, labelsize=FS_TICK)
    ax_gpu.tick_params(axis="y", labelsize=FS_TICK)
    ax_gpu.set_title("GPU metrics", fontsize=FS_AXIS, pad=6)

    # ── CPU panel ──────────────────────────────────────────────────────────────
    ax_cpu.plot(t_vals, cpu_line, color="#d62728", lw=2.2, zorder=3)
    ax_cpu.plot(t_vals, cpu_max,  color="#fc8d59", lw=1.8, ls="--", zorder=3)
    ax_cpu.set_ylim(-5, 115)
    ax_cpu.set_xlabel("Time (s from trace start)", fontsize=FS_AXIS)
    ax_cpu.set_ylabel("CPU %", fontsize=FS_AXIS)
    ax_cpu.tick_params(axis="both", labelsize=FS_TICK)
    ax_cpu.set_title("CPU metrics", fontsize=FS_AXIS, pad=6)

    # ── legend handles ─────────────────────────────────────────────────────────
    phase_handles = [
        mpatches.Patch(color="#1f77b4", alpha=0.5,  label="Generate (reasoning)"),
        mpatches.Patch(color="#aec7e8", alpha=0.5,  label="Generate (pre-loop)"),
        mpatches.Patch(color="#ff7f0e", alpha=0.65, label="Execute (tool call)"),
    ]
    gpu_handles = [
        plt.Line2D([0],[0], color="#2ca02c", lw=2.5, label=sm_label),
    ]
    if has_bw:
        gpu_handles.append(plt.Line2D([0],[0], color="#17becf", lw=2.0, ls="-.", label="DRAM activity % (HBM BW)"))
    if has_occ:
        gpu_handles.append(plt.Line2D([0],[0], color="#8c564b", lw=1.6, ls=":", label="SM occupancy % (low = BW-bound)"))
    if has_kv:
        gpu_handles.append(plt.Line2D([0],[0], color="#9467bd", lw=1.8, label="KV cache pool %"))

    cpu_handles = [
        plt.Line2D([0],[0], color="#d62728", lw=2.2, label=f"CPU mean (all {meta.get('n_cpu_sampled', 256)} cores)"),
        plt.Line2D([0],[0], color="#fc8d59", lw=1.8, ls="--", label="CPU hottest core"),
    ]

    # Place phase legend in upper legend area
    ax_leg_top.legend(handles=phase_handles + gpu_handles,
                      loc="upper left", fontsize=FS_LEGEND,
                      framealpha=0.9, edgecolor="#cccccc",
                      title="Phase shading & GPU lines", title_fontsize=FS_LEGEND - 1,
                      handlelength=2.5, handleheight=1.5, labelspacing=0.7)

    ax_leg_bot.legend(handles=cpu_handles,
                      loc="upper left", fontsize=FS_LEGEND,
                      framealpha=0.9, edgecolor="#cccccc",
                      title="CPU lines", title_fontsize=FS_LEGEND - 1,
                      handlelength=2.5, handleheight=1.5, labelspacing=0.7)

    task_id = meta.get("task_id") or folder.parent.name
    fig.suptitle(f"Biomni profiler — {task_id}  |  GPU bubble: 8.8% of wall time",
                 fontsize=FS_TITLE, fontweight="bold", y=0.995)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    make_slide_figure(TRACE_DIR, OUT_PATH)
