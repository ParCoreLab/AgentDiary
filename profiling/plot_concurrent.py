#!/usr/bin/env python3
"""
Rich multi-panel figure for a concurrent (N-tenant) session produced by the SHARED sampler.

Everything is aligned on ONE session timeline (that's what the shared L3+L4 sampler buys us):
  [1] single-agent reference (SAME time scale, for comparison)
  [2] per-agent phase Gantt (generate=GPU reasoning / execute=CPU tool)
  [3] overlap state strip (fill / gpu-shared / cpu-overlap / single)
  [4] GPU: SM activity, DRAM bandwidth, SM occupancy  (%)
  [5] KV-cache pool % + #running requests
  [6] CPU: node mean %, hot cores, SGLang-server %
  [7] serving quality: TTFT (s) + TPOT (ms) over time

Usage:
  python profiling/plot_concurrent.py --session results_mn5_multi/session_XXXX/ \
         [--reference results_mn5/gillespie_microbiome/<ts>/] [--out fig.png] [--task gillespie_microbiome]
"""
import argparse, json, glob, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C_GEN = "#3B6EA5"; C_EXE = "#E08A1E"                       # generate / execute
C_FILL = "#4CA96B"; C_GPU = "#8FB8DE"; C_CPU = "#C65B2E"; C_ONE = "#E3E3E3"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", default="documentation/figs/fig_concurrent_full.png")
    ap.add_argument("--task", default="gillespie_microbiome")
    a = ap.parse_args()

    # ── load agents ──
    agents = []
    for ad in sorted(glob.glob(str(Path(a.session) / f"agent_*/{a.task}/*/"))):
        ad = Path(ad)
        meta = json.loads((ad / "meta.json").read_text())
        g, e = phase_spans(ad / "events.jsonl")
        name = next(p for p in ad.parts if p.startswith("agent_")).replace(f"_{a.task}", "")
        agents.append(dict(dir=ad, t0=meta["t0_wall_unix"], name=name, g=g, e=e))
    if not agents:
        raise SystemExit(f"no agents found under {a.session}")

    # canonical timeline = earliest agent's reprojected CSVs (whole session, in its frame)
    canon = min(agents, key=lambda x: x["t0"])
    hw = load_csv(canon["dir"] / "hardware.csv")
    sg = load_csv(canon["dir"] / "sglang_metrics.csv")
    hw_wall = canon["t0"] + hw["t"]; sg_wall = canon["t0"] + sg["t"]
    ev_walls = [ag["t0"] + s for ag in agents for s, _ in ag["g"] + ag["e"]]
    origin = min([hw_wall.min()] + ev_walls)
    hw_t = hw_wall - origin; sg_t = sg_wall - origin
    for ag in agents:
        sh = ag["t0"] - origin
        ag["gs"] = [(s + sh, en + sh) for s, en in ag["g"]]
        ag["es"] = [(s + sh, en + sh) for s, en in ag["e"]]
    xmax = max([en for ag in agents for _, en in ag["gs"] + ag["es"]] + [hw_t.max()])

    # ── reference single-agent: DETERMINISTIC = the completed run closest to the MEDIAN wall,
    #    so it is representative (single-agent generation length is highly stochastic) ──
    sa_min = sa_med = sa_max = None
    if a.reference is None:
        cands = []
        for d in sorted(glob.glob(f"results_mn5/{a.task}/*/analysis.json")):
            an = json.load(open(d))
            if an["meta"].get("error") is None and (an["timing"].get("execute_total_s") or 0) > 5:
                cands.append((an["meta"]["wall_time_s"], str(Path(d).parent)))
        if cands:
            ws = [w for w, _ in cands]
            sa_min, sa_med, sa_max = min(ws), float(np.median(ws)), max(ws)
            a.reference = min(cands, key=lambda x: abs(x[0] - sa_med))[1]   # closest to median
    try:
        ref_wall = json.load(open(Path(a.reference) / "analysis.json"))["meta"]["wall_time_s"]
    except Exception:
        ref_wall = None
    rg, re_ = phase_spans(Path(a.reference) / "events.jsonl")
    r0 = min([s for s, _ in rg + re_])
    rg = [(s - r0, e - r0) for s, e in rg]; re_ = [(s - r0, e - r0) for s, e in re_]

    # ── overlap strip ──
    dt = 0.25; T = np.arange(0, xmax + dt, dt); strip = []
    for t in T:
        ng = sum(any(s <= t < e for s, e in ag["gs"]) for ag in agents)
        ne = sum(any(s <= t < e for s, e in ag["es"]) for ag in agents)
        strip.append(C_FILL if (ng and ne) else C_GPU if ng >= 2 else C_CPU if ne >= 2 else C_ONE)

    # ================= FIGURE =================
    N = len(agents)
    heights = [0.9, 0.28 * N + 0.5, 0.34, 1.15, 1.15, 1.15, 1.2]
    fig = plt.figure(figsize=(15, sum(heights) * 1.05 + 1.2))
    gs = fig.add_gridspec(7, 1, height_ratios=heights, hspace=0.55)
    fig.suptitle(f"Biomni {a.task} — single run (reference) vs. {N} concurrent tenants on one server",
                 fontsize=16, fontweight="bold", y=0.998)

    def stylex(ax, last=False):
        ax.set_xlim(0, xmax * 1.01); ax.grid(axis="x", alpha=0.25)
        ax.tick_params(labelbottom=last)   # only the bottom panel shows x labels (shared axis)

    # [1] reference (same scale)
    ax = fig.add_subplot(gs[0])
    ax.broken_barh([(s, e - s) for s, e in rg], (0.3, 0.4), facecolors=C_GEN)
    ax.broken_barh([(s, e - s) for s, e in re_], (0.3, 0.4), facecolors=C_EXE)
    ax.set_ylim(0, 1); ax.set_yticks([0.5]); ax.set_yticklabels(["1 agent"]); stylex(ax)
    note = (f"   [this run {ref_wall:.0f}s; single-agent varies {sa_min:.0f}–{sa_max:.0f}s, median {sa_med:.0f}s]"
            if (ref_wall and sa_med) else "")
    ax.set_title("① Single agent, alone (reference, same scale): GPU-reasoning / CPU-tool phases strictly alternate" + note,
                 fontsize=10.5, loc="left")

    # [2] concurrent Gantt
    axG = fig.add_subplot(gs[1], sharex=ax)
    for i, ag in enumerate(agents):
        y = N - 1 - i
        axG.broken_barh([(s, en - s) for s, en in ag["gs"]], (y + 0.12, 0.76), facecolors=C_GEN)
        axG.broken_barh([(s, en - s) for s, en in ag["es"]], (y + 0.12, 0.76), facecolors=C_EXE)
    axG.set_ylim(0, N); axG.set_yticks([N - 1 - i + 0.5 for i in range(N)])
    axG.set_yticklabels([ag["name"] for ag in agents], fontsize=8); stylex(axG)
    axG.set_title(f"② {N} agents concurrent — phases interleave (one agent's CPU tool overlaps another's GPU reasoning)",
                  fontsize=10.5, loc="left")

    # [3] overlap strip
    axO = fig.add_subplot(gs[2], sharex=ax)
    axO.bar(T, [1] * len(T), width=dt, color=strip, align="edge", linewidth=0)
    axO.set_ylim(0, 1); axO.set_yticks([0.5]); axO.set_yticklabels(["overlap"]); stylex(axO)

    # [4] GPU
    axH = fig.add_subplot(gs[3], sharex=ax)
    for suf, lab, c in [("sm_act", "SM activity", "#2a9d3f"), ("dram_act", "DRAM bw", "#17becf"),
                        ("sm_occ", "SM occupancy", "#8B4513")]:
        m = gpu_mean(hw, suf)
        if m is not None: axH.plot(hw_t, m, label=lab, lw=1.1, color=c)
    axH.set_ylabel("GPU %"); axH.legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.6); stylex(axH)
    axH.set_title("④ GPU (mean of 4×H100): SM activity high + SM occupancy low ⇒ memory-bandwidth-bound decode",
                  fontsize=10, loc="left")

    # [5] KV cache % + running reqs
    axK = fig.add_subplot(gs[4], sharex=ax)
    if "token_usage" in sg: axK.plot(sg_t, sg["token_usage"] * 100, color="#7b2fa0", lw=1.3, label="KV cache used %")
    axK.set_ylabel("KV %", color="#7b2fa0"); axK.set_ylim(0, max(5, np.nanmax(sg.get("token_usage", [0])) * 110))
    axK2 = axK.twinx()
    if "num_running_reqs" in sg: axK2.plot(sg_t, sg["num_running_reqs"], color="#d62728", lw=1.1, label="#running reqs")
    if "num_queue_reqs" in sg: axK2.plot(sg_t, sg["num_queue_reqs"], color="#ff7f0e", lw=0.9, ls="--", label="#queued")
    axK2.set_ylabel("requests", color="#d62728"); stylex(axK)
    axK.set_title("⑤ KV-cache headroom + concurrency (how many requests the server is actually running)", fontsize=10, loc="left")
    l1, la1 = axK.get_legend_handles_labels(); l2, la2 = axK2.get_legend_handles_labels()
    axK.legend(l1 + l2, la1 + la2, loc="upper right", ncol=3, fontsize=8, framealpha=0.6)

    # [6] CPU — node mean/max on 0-100, hot-core count on the right (is the CPU near bound?)
    axC = fig.add_subplot(gs[5], sharex=ax)
    if "cpu_max" in hw:  axC.plot(hw_t, hw["cpu_max"], color="#e0872f", lw=0.8, alpha=0.7, label="hottest core %")
    if "cpu_mean" in hw: axC.plot(hw_t, hw["cpu_mean"], color="#c1272d", lw=1.3, label="node mean % (all cores)")
    axC.set_ylabel("CPU %"); axC.set_ylim(0, 105)
    axC2 = axC.twinx()
    if "cpu_hot_cores" in hw:
        axC2.plot(hw_t, hw["cpu_hot_cores"], color="#333", lw=1.0, ls=":", label="#cores >50%")
        axC2.set_ylim(0, max(4, np.nanmax(hw["cpu_hot_cores"]) * 1.3))
    axC2.set_ylabel("hot cores"); stylex(axC)
    l1, la1 = axC.get_legend_handles_labels(); l2, la2 = axC2.get_legend_handles_labels()
    axC.legend(l1 + l2, la1 + la2, loc="upper left", ncol=3, fontsize=8, framealpha=0.6)
    axC.set_title("⑥ CPU (node, 80 cores): hottest core near 100% but node-mean ~2% and only a few hot cores — nowhere near bound",
                  fontsize=10, loc="left")

    # [7] TTFT / TPOT
    axT = fig.add_subplot(gs[6], sharex=ax)
    (tt, tv), (pt, pv) = (rate_series(sg, sg_t, "ttft_sum", "ttft_count", 1.0),
                          rate_series(sg, sg_t, "tpot_sum", "tpot_count", 1000.0))
    if len(tt): axT.plot(tt, tv, ".", ms=3, color="#c1272d", label="TTFT (s)")
    axT.set_ylabel("TTFT (s)", color="#c1272d"); stylex(axT, last=True)
    axT2 = axT.twinx()
    if len(pt): axT2.plot(pt, pv, ".", ms=3, color="#2a9d3f", label="TPOT (ms)")
    axT2.set_ylabel("TPOT (ms)", color="#2a9d3f")
    axT.set_title("⑦ Serving quality over time: TTFT (time-to-first-token) and TPOT (per-token) per LLM call", fontsize=10, loc="left")
    axT.set_xlabel("seconds from session start")

    leg = [Patch(fc=C_GEN, label="generate (GPU reasoning)"), Patch(fc=C_EXE, label="execute (CPU tool)"),
           Patch(fc=C_FILL, label="FILL: GPU busy while ≥1 tool runs"), Patch(fc=C_GPU, label="≥2 reasoning (GPU shared)"),
           Patch(fc=C_CPU, label="≥2 tools (CPU overlap)"), Patch(fc=C_ONE, label="≤1 active")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=6, fontsize=9, frameon=False)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=140, bbox_inches="tight")
    from collections import Counter
    cc = Counter(strip); tot = len(strip)
    print(f"saved {a.out}")
    print(f"session {xmax:.0f}s | agents {N} | fill {100*cc[C_FILL]/tot:.0f}% | gpu-shared {100*cc[C_GPU]/tot:.0f}%"
          f" | cpu-overlap {100*cc[C_CPU]/tot:.0f}% | single {100*cc[C_ONE]/tot:.0f}%")
    print(f"reference: {Path(a.reference).name}")


if __name__ == "__main__":
    main()
