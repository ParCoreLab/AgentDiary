#!/usr/bin/env python3
"""
Rich multi-panel figure for a SUSTAINED-load session (run_sustained.py): ~N concurrent agents drawing
a RANDOM task from a mix, shared sampler, session-level CSVs. Same visual language as plot_concurrent.py
but (a) multi-task (Gantt grouped by task family) and (b) reads session_hardware.csv / session_sglang_metrics.csv
directly (no per-agent reprojected CSVs — the sustained launcher only keeps the session-level traces).

Everything is aligned on ONE session clock: session CSV t=0 == sampler start (sampler_t0_wall_unix in
session_summary.json). Each agent's events.jsonl (its own perf_counter frame) is shifted onto the session
clock by (agent meta t0_wall_unix - sampler_t0_wall_unix).

  [1] per-agent phase Gantt, grouped by task (generate=GPU reasoning / execute=CPU tool)
  [2] overlap state strip (fill / gpu-shared / cpu-overlap / single)
  [3] GPU: SM activity, DRAM bandwidth, SM occupancy (%)
  [4] KV-cache pool % + #running / #queued requests
  [5] CPU: node mean %, hottest core %, #hot cores
  [6] serving quality: TTFT (s) + TPOT (ms) per LLM call

  python profiling/plot_sustained.py --session results_mn5_multi/session_XXXX/ --out documentation/figs/fig_concurrent_sustained40.png
"""
import argparse, json, glob, csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C_GEN = "#3B6EA5"; C_EXE = "#E08A1E"                                # generate / execute
C_FILL = "#4CA96B"; C_GPU = "#8FB8DE"; C_CPU = "#C65B2E"; C_ONE = "#E3E3E3"
TASK_C = ["#6C5B9E", "#2E8B7A", "#B5533C", "#4C78A8", "#8A6D3B", "#933B6B"]  # per-task left band
NCPU_DEFAULT = 160


def load_csv(path):
    with open(path) as f:
        r = csv.reader(f); hdr = next(r)
        cols = {h: [] for h in hdr}
        for row in r:
            for h, v in zip(hdr, row):
                try: cols[h].append(float(v))
                except (ValueError, TypeError): cols[h].append(np.nan)
    return {h: np.array(v, dtype=float) for h, v in cols.items()}


def phase_spans(evpath):
    gens, exes, og, oe = [], [], [], []
    for ln in open(evpath):
        try: e = json.loads(ln)
        except Exception: continue
        ph, ed, t = e.get("phase"), e.get("edge"), e.get("t")
        if t is None: continue
        if ph == "generate":
            og.append(t) if ed == "start" else (gens.append((og.pop(), t)) if og else None)
        elif ph == "execute":
            oe.append(t) if ed == "start" else (exes.append((oe.pop(), t)) if oe else None)
    return gens, exes


def gpu_mean(hw, suffix):
    cols = [hw[f"gpu{i}_{suffix}"] for i in range(8) if f"gpu{i}_{suffix}" in hw]
    cols = [c for c in cols if not np.all(np.isnan(c))]
    return np.nanmean(np.vstack(cols), axis=0) if cols else None


def rate_series(sg, t, sumk, cntk, scale):
    s, c = sg.get(sumk), sg.get(cntk)
    if s is None or c is None: return np.array([]), np.array([])
    ts, ys = [], []
    for i in range(1, len(c)):
        dc, ds = c[i] - c[i - 1], s[i] - s[i - 1]
        if dc > 0 and ds >= 0:
            ts.append(t[i]); ys.append(ds / dc * scale)
    return np.array(ts), np.array(ys)


def concurrency(spans_per_agent, grid):
    """Step-function count of agents whose span covers each grid point (via +1/-1 sweep)."""
    ev = []
    for spans in spans_per_agent:
        for s, e in spans:
            ev.append((s, 1)); ev.append((e, -1))
    if not ev:
        return np.zeros_like(grid)
    ev.sort()
    ts = np.array([x[0] for x in ev]); dv = np.cumsum([x[1] for x in ev])
    idx = np.searchsorted(ts, grid, side="right") - 1
    out = np.where(idx >= 0, dv[np.clip(idx, 0, len(dv) - 1)], 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ncpu", type=int, default=NCPU_DEFAULT, help="logical cores on the node (for context lines)")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    sess = Path(a.session)
    out = a.out or str(sess / "fig_concurrent_sustained.png")

    summ = json.loads((sess / "session_summary.json").read_text())
    origin_wall = summ["sampler_t0_wall_unix"]           # session CSV t=0 corresponds to this wall time
    target = summ.get("target"); launch_win = summ.get("duration_s")

    # ── load agents (any task subdir) ──
    agents = []
    metas = (sorted(glob.glob(str(sess / "agents/agent_*/*/*/meta.json")))       # organized layout
             or sorted(glob.glob(str(sess / "agent_*/*/*/meta.json"))))          # legacy flat layout
    for mp in metas:
        ad = Path(mp).parent
        try:
            meta = json.loads((ad / "meta.json").read_text())
            g, e = phase_spans(ad / "events.jsonl")
        except Exception:
            continue
        adir = next(p for p in ad.parts if p.startswith("agent_"))
        task = ad.parts[-2]                              # <task_id>/<ts>
        shift = meta["t0_wall_unix"] - origin_wall
        agents.append(dict(name=adir, task=task, t0=shift,
                           gs=[(s + shift, en + shift) for s, en in g],
                           es=[(s + shift, en + shift) for s, en in e],
                           err=meta.get("error")))
    if not agents:
        raise SystemExit(f"no agents with events found under {sess}")
    N = len(agents)

    # ── session CSVs (already on session clock) ──
    hw = load_csv(sess / "session_hardware.csv"); sg = load_csv(sess / "session_sglang_metrics.csv")
    hw_t, sg_t = hw["t"], sg["t"]
    xmax = max(hw_t.max(), max((en for ag in agents for _, en in ag["gs"] + ag["es"]), default=0))

    # ── group agents by task, sort by arrival within group ──
    tasks = sorted({ag["task"] for ag in agents})
    tcolor = {t: TASK_C[i % len(TASK_C)] for i, t in enumerate(tasks)}
    order = []
    for t in tasks:
        order += sorted([ag for ag in agents if ag["task"] == t], key=lambda x: x["t0"])
    for y, ag in enumerate(order):
        ag["y"] = N - 1 - y                               # top = first task group

    # ── overlap strip via concurrency sweep ──
    dt = 0.5; grid = np.arange(0, xmax + dt, dt)
    ng = concurrency([ag["gs"] for ag in agents], grid)
    ne = concurrency([ag["es"] for ag in agents], grid)
    strip = [C_FILL if (g and e) else C_GPU if g >= 2 else C_CPU if e >= 2 else C_ONE
             for g, e in zip(ng, ne)]

    # ================= FIGURE =================
    heights = [0.135 * N + 0.6, 0.34, 1.15, 1.15, 1.15, 1.15]
    H = sum(heights) * 1.02 + 1.6
    fig = plt.figure(figsize=(15, H))
    # reserve a constant ~0.66in title band regardless of figure height so suptitle never collides with panel ①
    gsp = fig.add_gridspec(6, 1, height_ratios=heights, hspace=0.42,
                           top=1 - 0.66 / H, bottom=0.045, left=0.13, right=0.93)
    title = a.title or (f"Biomni sustained load — ~{target} concurrent agents, {len(tasks)}-task CPU-heavy mix "
                        f"({N} launched) on one server")
    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=1 - 0.32 / H)

    def stylex(ax, last=False):
        ax.set_xlim(0, xmax * 1.005); ax.grid(axis="x", alpha=0.25)
        ax.tick_params(labelbottom=last)
        if launch_win:
            ax.axvspan(0, launch_win, color="#F3EDE1", zorder=0)

    # [1] Gantt grouped by task
    ax = fig.add_subplot(gsp[0])
    for ag in agents:
        y = ag["y"]
        ax.broken_barh([(s, en - s) for s, en in ag["gs"]], (y + 0.1, 0.8), facecolors=C_GEN)
        ax.broken_barh([(s, en - s) for s, en in ag["es"]], (y + 0.1, 0.8), facecolors=C_EXE)
        ax.add_patch(plt.Rectangle((-0.012 * xmax, y + 0.1), 0.008 * xmax, 0.8,
                                   facecolor=tcolor[ag["task"]], clip_on=False, lw=0))
    # task-group labels + separators
    yt, ytl = [], []
    for t in tasks:
        ys = [ag["y"] for ag in agents if ag["task"] == t]
        yt.append(np.mean(ys) + 0.5); ytl.append(f"{t}\n(n={len(ys)})")
        ax.axhline(min(ys), color="#CCC", lw=0.6)
    ax.set_yticks(yt); ax.set_yticklabels(ytl, fontsize=8)
    for tick, t in zip(ax.get_yticklabels(), tasks): tick.set_color(tcolor[t])
    ax.set_ylim(0, N); stylex(ax)
    if launch_win:
        ax.axvline(launch_win, color="#9a8a6a", ls="--", lw=1.0)
        ax.text(launch_win, N, "  launch window ends → drain", va="top", ha="left", fontsize=8.5,
                color="#7A6A50", style="italic")
    ax.set_title(f"① {N} agents — per-agent phase Gantt grouped by task (generate=GPU reasoning / execute=CPU tool)",
                 fontsize=10.5, loc="left")

    # [2] overlap strip
    axO = fig.add_subplot(gsp[1], sharex=ax)
    axO.bar(grid, [1] * len(grid), width=dt, color=strip, align="edge", linewidth=0)
    axO.set_ylim(0, 1); axO.set_yticks([0.5]); axO.set_yticklabels(["overlap"]); stylex(axO)

    # [3] GPU
    axH = fig.add_subplot(gsp[2], sharex=ax)
    for suf, lab, c in [("sm_act", "SM activity", "#2a9d3f"), ("dram_act", "DRAM bw", "#17becf"),
                        ("sm_occ", "SM occupancy", "#8B4513")]:
        m = gpu_mean(hw, suf)
        if m is not None: axH.plot(hw_t, m, label=lab, lw=1.0, color=c)
    axH.set_ylabel("GPU %"); axH.set_ylim(0, 100)
    axH.legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.6); stylex(axH)
    axH.set_title("③ GPU (mean of 4): SM activity high but SM occupancy low ⇒ memory-bandwidth-bound decode",
                  fontsize=10, loc="left")

    # [4] KV cache % + running/queued
    axK = fig.add_subplot(gsp[3], sharex=ax)
    if "token_usage" in sg:
        axK.plot(sg_t, sg["token_usage"] * 100, color="#7b2fa0", lw=1.3, label="KV cache used %")
    axK.set_ylabel("KV %", color="#7b2fa0"); axK.set_ylim(0, 100)
    axK2 = axK.twinx()
    if "num_running_reqs" in sg: axK2.plot(sg_t, sg["num_running_reqs"], color="#d62728", lw=1.0, label="#running reqs")
    if "num_queue_reqs" in sg: axK2.plot(sg_t, sg["num_queue_reqs"], color="#ff7f0e", lw=1.0, ls="--", label="#queued")
    axK2.set_ylabel("requests", color="#d62728"); stylex(axK)
    axK.set_title("④ KV-cache headroom + concurrency actually served (running / queued requests)", fontsize=10, loc="left")
    l1, la1 = axK.get_legend_handles_labels(); l2, la2 = axK2.get_legend_handles_labels()
    axK.legend(l1 + l2, la1 + la2, loc="upper left", ncol=3, fontsize=8, framealpha=0.6)

    # [5] CPU
    axC = fig.add_subplot(gsp[4], sharex=ax)
    if "cpu_max" in hw:  axC.plot(hw_t, hw["cpu_max"], color="#e0872f", lw=0.7, alpha=0.6, label="hottest core %")
    if "cpu_mean" in hw: axC.plot(hw_t, hw["cpu_mean"], color="#c1272d", lw=1.3, label="node mean % (all cores)")
    axC.set_ylabel("CPU %"); axC.set_ylim(0, 105)
    axC2 = axC.twinx()
    if "cpu_hot_cores" in hw:
        axC2.plot(hw_t, hw["cpu_hot_cores"], color="#333", lw=1.0, ls=":", label="#cores >50%")
        axC2.set_ylim(0, a.ncpu)
        axC2.axhline(a.ncpu, color="#999", ls=":", lw=0.7)
    axC2.set_ylabel(f"hot cores (of {a.ncpu})"); stylex(axC)
    l1, la1 = axC.get_legend_handles_labels(); l2, la2 = axC2.get_legend_handles_labels()
    axC.legend(l1 + l2, la1 + la2, loc="upper right", ncol=3, fontsize=8, framealpha=0.6)
    pk = np.nanmax(hw["cpu_mean"]) if "cpu_mean" in hw else 0
    axC.set_title(f"⑤ CPU (node, {a.ncpu} logical cores): node-mean peaks {pk:.0f}% "
                  f"(~{pk*a.ncpu/100:.0f} busy cores) — is the CPU approaching bound?", fontsize=10, loc="left")

    # [6] TTFT / TPOT
    axT = fig.add_subplot(gsp[5], sharex=ax)
    (tt, tv) = rate_series(sg, sg_t, "ttft_sum", "ttft_count", 1.0)
    (pt, pv) = rate_series(sg, sg_t, "tpot_sum", "tpot_count", 1000.0)
    if len(tt): axT.plot(tt, tv, ".", ms=2.5, color="#c1272d", label="TTFT (s)")
    axT.set_ylabel("TTFT (s)", color="#c1272d"); stylex(axT, last=True)
    axT2 = axT.twinx()
    if len(pt): axT2.plot(pt, pv, ".", ms=2.5, color="#2a9d3f", label="TPOT (ms)")
    axT2.set_ylabel("TPOT (ms)", color="#2a9d3f")
    axT.set_title("⑥ Serving quality over time: TTFT (time-to-first-token) and TPOT (per-token) per LLM call",
                  fontsize=10, loc="left")
    axT.set_xlabel("seconds from session start")

    leg = [Patch(fc=C_GEN, label="generate (GPU reasoning)"), Patch(fc=C_EXE, label="execute (CPU tool)"),
           Patch(fc=C_FILL, label="FILL: GPU busy while ≥1 tool runs"), Patch(fc=C_GPU, label="≥2 reasoning (GPU shared)"),
           Patch(fc=C_CPU, label="≥2 tools (CPU overlap)"), Patch(fc=C_ONE, label="≤1 active")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.012), ncol=6, fontsize=9, frameon=False)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    cc = Counter(strip); tot = len(strip)
    print(f"saved {out}")
    print(f"session {xmax:.0f}s | agents {N} | tasks {tasks}")
    print(f"fill {100*cc[C_FILL]/tot:.0f}% | gpu-shared {100*cc[C_GPU]/tot:.0f}% | "
          f"cpu-overlap {100*cc[C_CPU]/tot:.0f}% | single {100*cc[C_ONE]/tot:.0f}%")


if __name__ == "__main__":
    main()
