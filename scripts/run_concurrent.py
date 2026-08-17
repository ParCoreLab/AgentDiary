#!/usr/bin/env python3
"""
run_concurrent.py — launch N Biomni agents concurrently against one shared SGLang server.

Usage:
    python scripts/run_concurrent.py session_configs/block_b1_fill.json
    python scripts/run_concurrent.py session_configs/block_b1_fill.json --results-root results_multi

Session config JSON format (see session_configs/ for examples):
    {
      "session_name":  "block_b1_fill",
      "description":   "Block B1: one CPU-heavy + one GPU-heavy agent, phases overlapping",
      "agents": [
        {
          "agent_id":         "agent_0",
          "task_config":      "tasks/random_forest_cv_large.json",
          "arrival_offset_s": 0,
          "numactl_node":     null      // null = no numactl; 0 or 1 for NUMA binding
        },
        {
          "agent_id":         "agent_1",
          "task_config":      "tasks/admet_ibuprofen.json",
          "arrival_offset_s": 60
        }
      ]
    }

Output tree:
    results_multi/session_<timestamp>/
        session_config.json            input config + resolved metadata
        agent_0/<task_id>/<timestamp>/ same structure as single-agent results/
        agent_1/...
        session_summary.json           cross-agent metrics + phase overlap analysis

Key design rules (from multi_agent_design_handoff.md):
  - Separate OS processes, not threads (the class-level generate hook is shared state)
  - numactl binding is OPTIONAL (stability only), never a swept variable by default
  - Staggered arrival offsets prevent synchronized startup burst (retrieval collision)
  - Per-agent analysis is already auto-run by trace_run.py before the process exits,
    so analysis.json and figures exist as soon as proc.wait() returns
"""

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "profiling"))
import trace_run as _tr  # shared L3+L4 sampler classes + the T0_MONO/T0_WALL wall-clock anchor


def _reproject_csv(src: Path, dst: Path, offset: float):
    """Copy a session CSV, shifting only its 't' column by `offset` into an agent's timeframe."""
    import csv
    if not src.exists():
        return
    with open(src, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    hdr = rows[0]
    try:
        ti = hdr.index("t")
    except ValueError:
        return
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in rows[1:]:
            if len(r) > ti:
                try:
                    r[ti] = f"{float(r[ti]) + offset:.4f}"
                except ValueError:
                    pass
            w.writerow(r)


def _reproject_and_analyze(trace_dir: Path, sess_hw: Path, sess_sg: Path, sampler_t0_wall: float):
    """Shift the shared session CSVs into this agent's perf-counter frame (so analyze_trace.py
    joins them against the agent's own events unchanged), then run analyze_trace.py on the folder."""
    try:
        meta = json.loads((trace_dir / "meta.json").read_text())
    except Exception:
        return
    agent_t0 = meta.get("t0_wall_unix")
    if agent_t0 is None:
        return
    offset = sampler_t0_wall - agent_t0    # session-frame t + offset -> this agent's frame
    _reproject_csv(sess_hw, trace_dir / "hardware.csv", offset)
    _reproject_csv(sess_sg, trace_dir / "sglang_metrics.csv", offset)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "profiling" / "analyze_trace.py"), str(trace_dir)],
        check=False,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def find_trace_dir(agent_outdir: Path, task_id: str) -> Path | None:
    """Return the single trace directory written by trace_run.py for this agent."""
    task_dir = agent_outdir / task_id
    if not task_dir.exists():
        return None
    subdirs = sorted(d for d in task_dir.iterdir() if d.is_dir())
    return subdirs[-1] if subdirs else None


def load_json(path: Path) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_events_abs(trace_dir: Path) -> tuple[list, list]:
    """
    Parse events.jsonl and return (gen_spans, exe_spans) as absolute wall-clock
    intervals: [(abs_start, abs_end), ...].

    Absolute time = meta["t0_wall_unix"] + event["t"] (perf_counter offset).
    Each agent process has its own T0_WALL, so this is self-contained per agent.
    """
    meta = load_json(trace_dir / "meta.json")
    if meta is None:
        return [], []
    t0 = meta.get("t0_wall_unix", 0.0)

    events_path = trace_dir / "events.jsonl"
    if not events_path.exists():
        return [], []

    gen_spans, exe_spans = [], []
    open_gen = open_exe = None

    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t_abs = t0 + ev.get("t", 0.0)
            phase = ev.get("phase")
            edge  = ev.get("edge")
            if phase == "generate":
                if edge == "start":
                    open_gen = t_abs
                elif edge == "end" and open_gen is not None:
                    gen_spans.append((open_gen, t_abs))
                    open_gen = None
            elif phase == "execute":
                if edge == "start":
                    open_exe = t_abs
                elif edge == "end" and open_exe is not None:
                    exe_spans.append((open_exe, t_abs))
                    open_exe = None

    return gen_spans, exe_spans


def interval_overlap_total(spans_a: list, spans_b: list) -> float:
    """Total overlapping seconds between two lists of (start, end) intervals."""
    total = 0.0
    for (a0, a1) in spans_a:
        for (b0, b1) in spans_b:
            ov = min(a1, b1) - max(a0, b0)
            if ov > 0:
                total += ov
    return total


def compute_overlap_analysis(agent_infos: list, t_session_start: float,
                              t_session_end: float) -> dict:
    """
    Compute cross-agent phase overlap statistics.

    agent_infos: [{"gen_spans": [...], "exe_spans": [...]}, ...] with abs times.

    fill_s         — seconds where ≥1 agent is in generate while ≥1 OTHER is in execute.
                     This is the "free lunch": different agents using different resources.
    gpu_collision_s — seconds where ≥2 agents are both in generate (shared GPU batching).
    cpu_collision_s — seconds where ≥2 agents are both in execute (CPU/DDR contention).
    """
    n = len(agent_infos)
    session_wall_s = max(t_session_end - t_session_start, 1e-6)

    fill_s = gpu_collision_s = cpu_collision_s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = agent_infos[i], agent_infos[j]
            fill_s          += interval_overlap_total(a["gen_spans"], b["exe_spans"])
            fill_s          += interval_overlap_total(a["exe_spans"], b["gen_spans"])
            gpu_collision_s += interval_overlap_total(a["gen_spans"], b["gen_spans"])
            cpu_collision_s += interval_overlap_total(a["exe_spans"], b["exe_spans"])

    return {
        "session_wall_s":          round(session_wall_s, 3),
        "fill_s":                  round(fill_s, 3),
        "gpu_collision_s":         round(gpu_collision_s, 3),
        "cpu_collision_s":         round(cpu_collision_s, 3),
        "fill_fraction":           round(fill_s / session_wall_s, 4),
        "gpu_collision_fraction":  round(gpu_collision_s / session_wall_s, 4),
        "cpu_collision_fraction":  round(cpu_collision_s / session_wall_s, 4),
        "note": (
            "fill: ≥1 agent in generate while ≥1 other in execute — the bubble fill. "
            "gpu_collision: ≥2 agents both in generate — GPU batching / TPOT pressure. "
            "cpu_collision: ≥2 agents both in execute — CPU/DDR contention."
        ),
    }


def _get_task_id(agent_cfg: dict) -> str:
    try:
        with open(agent_cfg["task_config"]) as f:
            return json.load(f)["task_id"]
    except Exception:
        return Path(agent_cfg["task_config"]).stem


def _extract_agent_metrics(trace_dir: Path | None) -> dict:
    """Pull the key per-agent metrics from analysis.json and meta.json."""
    if trace_dir is None:
        return {"analysis_available": False}

    analysis = load_json(trace_dir / "analysis.json")
    meta      = load_json(trace_dir / "meta.json")

    if analysis is None:
        return {"analysis_available": False, "trace_dir": str(trace_dir)}

    bubble  = analysis.get("bubble", {})
    timing  = analysis.get("timing", {})
    turns   = analysis.get("turns", {})
    l2      = analysis.get("layer2", {})
    l3      = analysis.get("layer3", {})
    gpu_sat = analysis.get("gpu_saturation", {})
    cpu_ut  = analysis.get("cpu_utilization", {})

    return {
        "analysis_available":        True,
        "trace_dir":                 str(trace_dir),
        # bubble — the headline: does fill reduce it under concurrency?
        "gpu_bubble_weighted_frac":  bubble.get("gpu_weighted_fraction"),
        "gpu_bubble_binary_frac":    bubble.get("gpu_binary_fraction"),
        "cpu_bubble_weighted_frac":  bubble.get("cpu_weighted_fraction"),
        # timing
        "wall_time_s":               analysis.get("meta", {}).get("wall_time_s"),
        "generate_total_s":          timing.get("generate_total_s"),
        "execute_total_s":           timing.get("execute_total_s"),
        "n_real_turns":              turns.get("n_real_turns"),
        # L2 — token counts + latency
        "l2_n_llm_calls":            l2.get("n_llm_calls"),
        "l2_mean_latency_s":         l2.get("mean_latency_s"),
        "l2_mean_prompt_tokens":     l2.get("mean_prompt_tokens"),
        "l2_mean_completion_tokens": l2.get("mean_completion_tokens"),
        # L3 — SGLang latency + KV (the collision signals)
        "l3_mean_ttft_s":            l3.get("mean_ttft_s"),
        "l3_mean_tpot_ms":           l3.get("mean_tpot_ms"),
        "l3_mean_queue_time_ms":     l3.get("mean_queue_time_ms"),
        "l3_queue_wait_mean_ms":     l3.get("queue_wait_mean_ms"),
        "kv_cache_peak_pct":         l3.get("kv_peak_pct"),
        # GPU saturation split by phase
        "gpu_sm_util_generate":      gpu_sat.get("during_generate", {}).get("sm_util_mean"),
        "gpu_sm_util_execute":       gpu_sat.get("during_execute",  {}).get("sm_util_mean"),
        "gpu_mem_free_generate_mb":  gpu_sat.get("during_generate", {}).get("mem_free_mean_mb"),
        # CPU split by phase
        "cpu_mean_execute":          cpu_ut.get("during_execute", {}).get("mean"),
        # Per-process CPU baseline from the new Step-3 sampler
        "server_idle_cpu_baseline":  meta.get("server_idle_cpu_baseline") if meta else None,
        "server_n_procs":            meta.get("server_n_procs")           if meta else None,
        "error":                     analysis.get("meta", {}).get("error"),
    }


# ── baseline discovery ─────────────────────────────────────────────────────────

def find_single_agent_baseline(task_id: str) -> Path | None:
    """
    Find the most recent single-agent trace dir for task_id.
    Priority 1: results/<task_id>/  (standard non-multi-agent runs).
    Priority 2: results_multi/*/  sessions with n_agents == 1.
    Returns a trace dir containing analysis.json, or None.
    """
    std_dir = REPO_ROOT / "results" / task_id
    if std_dir.exists():
        trace_dirs = sorted(
            d for d in std_dir.iterdir()
            if d.is_dir() and (d / "analysis.json").exists()
        )
        if trace_dirs:
            return trace_dirs[-1]

    multi_root = REPO_ROOT / "results_multi"
    if multi_root.exists():
        candidates = []
        for sess_dir in multi_root.iterdir():
            if not sess_dir.is_dir():
                continue
            summ = load_json(sess_dir / "session_summary.json")
            if summ is None or summ.get("n_agents", 0) != 1:
                continue
            for agent_dir in sess_dir.iterdir():
                if not agent_dir.is_dir():
                    continue
                task_dir = agent_dir / task_id
                if not task_dir.exists():
                    continue
                trace_dirs = sorted(
                    d for d in task_dir.iterdir()
                    if d.is_dir() and (d / "analysis.json").exists()
                )
                if trace_dirs:
                    candidates.append(trace_dirs[-1])
        if candidates:
            return sorted(candidates)[-1]

    return None


# ── comparison report helpers ──────────────────────────────────────────────────

def _v(val, fmt=".1f", unit="", none_str="n/a") -> str:
    """Format a single number for table display."""
    return none_str if val is None else f"{val:{fmt}}{unit}"


def _ds(b, m) -> str:
    """Delta for seconds: ±X.Xs (±Y%)"""
    if b is None or m is None:
        return "—"
    d = m - b
    pct = d / b * 100 if b else 0.0
    return f"{d:+.1f}s ({pct:+.0f}%)"


def _dpp(b_pct, m_pct, *, reduce=False, fill=False, threshold=2.0) -> str:
    """Delta for percentage-point values with optional semantic annotation."""
    if b_pct is None or m_pct is None:
        return "—"
    d = m_pct - b_pct
    s = f"{d:+.1f}pp"
    if abs(d) >= threshold:
        if reduce and d < 0:
            s += "  ↓ REDUCED"
        elif reduce and d > 0:
            s += "  ↑ WORSENED"
        elif fill and d > 0:
            s += "  ★ FILL"
    return s


def _dms(b_ms, m_ms, safe_threshold_pct=10.0) -> str:
    """Delta for ms values with SAFE / DEGRADED annotation for inference quality."""
    if b_ms is None or m_ms is None:
        return "—"
    d = m_ms - b_ms
    pct = d / b_ms * 100 if b_ms else 0.0
    s = f"{d:+.2f}ms ({pct:+.0f}%)"
    if abs(pct) < safe_threshold_pct:
        s += "  ✓ SAFE"
    elif pct > safe_threshold_pct:
        s += "  ⚠ DEGRADED"
    return s


def build_comparison_report(session_name: str, agent_summaries: list,
                             baselines: dict) -> str:
    """
    Build a human-readable single-agent vs multi-agent comparison table.

    baselines: {task_id: metrics_dict} from _extract_agent_metrics().
    Returns a multiline string — print it and/or write to session_comparison.txt.
    """
    W   = 82
    SEP = "═" * W

    lines: list[str] = [
        SEP,
        f"  Single-agent vs Multi-agent Comparison — {session_name}",
        SEP,
    ]

    def row(label: str, b_str: str, m_str: str, d_str: str) -> None:
        lines.append(f"    {label:<30} {b_str:>13} {m_str:>13}   {d_str}")

    for s in agent_summaries:
        task_id = s["task_id"]
        b       = baselines.get(task_id) or {}

        hdr = f"  ── {s['agent_id']}  ·  {task_id} "
        lines += ["", hdr + "─" * max(0, W - len(hdr))]
        bdir = b.get("trace_dir")
        lines.append(f"  Baseline: {bdir}" if bdir
                     else "  Baseline: NOT FOUND — no single-agent trace for this task")
        lines += [
            "",
            f"  {'Metric':<34} {'Single-agent':>13} {'Multi-agent':>13}   Δ",
            "  " + "─" * (W - 2),
        ]

        # ── Timing ──────────────────────────────────────────────────────────
        lines.append("  ▸ Timing")
        row("wall_time_s",
            _v(b.get("wall_time_s")),      _v(s.get("wall_time_s")),
            _ds(b.get("wall_time_s"),       s.get("wall_time_s")))
        row("generate_total_s",
            _v(b.get("generate_total_s")), _v(s.get("generate_total_s")),
            _ds(b.get("generate_total_s"),  s.get("generate_total_s")))
        row("execute_total_s",
            _v(b.get("execute_total_s")),  _v(s.get("execute_total_s")),
            _ds(b.get("execute_total_s"),   s.get("execute_total_s")))
        n_b, n_m = b.get("n_real_turns"), s.get("n_real_turns")
        row("real_turns",
            _v(n_b, "d"),                  _v(n_m, "d"),
            "=" if n_b == n_m
            else (f"{n_m - n_b:+d}" if (n_b is not None and n_m is not None)
                  else "—"))

        # ── Bubble ──────────────────────────────────────────────────────────
        lines.append("  ▸ Bubble (GPU idle fraction)")
        b_wb = b.get("gpu_bubble_weighted_frac")
        m_wb = s.get("gpu_bubble_weighted_frac")
        b_bb = b.get("gpu_bubble_binary_frac")
        m_bb = s.get("gpu_bubble_binary_frac")
        row("gpu_bubble_weighted (%)",
            _v(None if b_wb is None else b_wb * 100, ".1f", "%"),
            _v(None if m_wb is None else m_wb * 100, ".1f", "%"),
            _dpp(None if b_wb is None else b_wb * 100,
                 None if m_wb is None else m_wb * 100,
                 reduce=True, threshold=2.0))
        row("gpu_bubble_binary (%)",
            _v(None if b_bb is None else b_bb * 100, ".1f", "%"),
            _v(None if m_bb is None else m_bb * 100, ".1f", "%"),
            _dpp(None if b_bb is None else b_bb * 100,
                 None if m_bb is None else m_bb * 100,
                 reduce=True, threshold=2.0))

        # ── GPU by phase ─────────────────────────────────────────────────────
        lines.append("  ▸ GPU utilization by phase")
        row("SM util during generate (%)",
            _v(b.get("gpu_sm_util_generate"), ".1f", "%"),
            _v(s.get("gpu_sm_util_generate"), ".1f", "%"),
            _dpp(b.get("gpu_sm_util_generate"), s.get("gpu_sm_util_generate"),
                 threshold=5.0))
        row("SM util during execute (%)",
            _v(b.get("gpu_sm_util_execute"), ".1f", "%"),
            _v(s.get("gpu_sm_util_execute"), ".1f", "%"),
            _dpp(b.get("gpu_sm_util_execute"), s.get("gpu_sm_util_execute"),
                 fill=True, threshold=5.0))

        # ── L2 ───────────────────────────────────────────────────────────────
        lines.append("  ▸ LLM calls (Layer-2)")
        lc_b, lc_m = b.get("l2_n_llm_calls"), s.get("l2_n_llm_calls")
        row("n_llm_calls",
            _v(lc_b, "d"), _v(lc_m, "d"),
            "=" if lc_b == lc_m
            else (f"{lc_m - lc_b:+d}" if (lc_b is not None and lc_m is not None)
                  else "—"))

        lat_b, lat_m = b.get("l2_mean_latency_s"), s.get("l2_mean_latency_s")
        d_lat = _ds(lat_b, lat_m)
        if lat_b and lat_m and (lat_m - lat_b) / lat_b > 0.5:
            d_lat += "  ⚠ LOAD"
        row("mean_latency_s", _v(lat_b, ".2f"), _v(lat_m, ".2f"), d_lat)

        pt_b, pt_m = b.get("l2_mean_prompt_tokens"), s.get("l2_mean_prompt_tokens")
        row("mean_prompt_tokens",
            _v(pt_b, ".0f"), _v(pt_m, ".0f"),
            (f"{pt_m - pt_b:+.0f} ({(pt_m - pt_b) / pt_b * 100:+.0f}%)"
             if (pt_b and pt_m) else "—"))

        ct_b, ct_m = (b.get("l2_mean_completion_tokens"),
                      s.get("l2_mean_completion_tokens"))
        row("mean_compl_tokens",
            _v(ct_b, ".0f"), _v(ct_m, ".0f"),
            (f"{ct_m - ct_b:+.0f} ({(ct_m - ct_b) / ct_b * 100:+.0f}%)"
             if (ct_b and ct_m) else "—"))

        # ── L3 ───────────────────────────────────────────────────────────────
        lines.append("  ▸ Inference quality (Layer-3)  ← key collision signals")
        ttft_b, ttft_m = b.get("l3_mean_ttft_s"), s.get("l3_mean_ttft_s")
        row("TTFT (s)",
            _v(ttft_b, ".3f"), _v(ttft_m, ".3f"),
            _dms(ttft_b * 1000 if ttft_b is not None else None,
                 ttft_m * 1000 if ttft_m is not None else None))

        tpot_b, tpot_m = b.get("l3_mean_tpot_ms"), s.get("l3_mean_tpot_ms")
        row("TPOT (ms)",
            _v(tpot_b, ".2f", "ms"), _v(tpot_m, ".2f", "ms"),
            _dms(tpot_b, tpot_m))

        qw_b, qw_m = b.get("l3_queue_wait_mean_ms"), s.get("l3_queue_wait_mean_ms")
        row("queue_wait (ms)",
            _v(qw_b, ".1f", "ms"), _v(qw_m, ".1f", "ms"),
            (f"{qw_m - qw_b:+.1f}ms"
             if (qw_b is not None and qw_m is not None) else "—"))

    lines += ["", SEP]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Launch N concurrent Biomni agents and produce a session summary.")
    ap.add_argument("session_config",
                    help="Path to session config JSON (see session_configs/ for examples).")
    ap.add_argument("--results-root", default="results_multi",
                    help="Root directory for multi-agent results (default: results_multi/).")
    ap.add_argument("--base-url", default="http://localhost:30000/v1")
    ap.add_argument("--hw-interval", type=float, default=0.05)
    args = ap.parse_args()

    cfg_path = Path(args.session_config)
    with open(cfg_path) as f:
        cfg = json.load(f)

    session_id  = ("session_" +
                   dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f"))
    session_dir = Path(args.results_root) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    n_agents = len(cfg["agents"])
    print(f"[concurrent] session:  {session_id}")
    print(f"[concurrent] output:   {session_dir}")
    print(f"[concurrent] agents:   {n_agents}")
    print(f"[concurrent] name:     {cfg.get('session_name','')}")

    # Persist session config alongside results
    session_config_out = {
        "session_id":    session_id,
        "session_name":  cfg.get("session_name", ""),
        "description":   cfg.get("description", ""),
        "source_config": str(cfg_path.resolve()),
        "session_dir":   str(session_dir.resolve()),
        "base_url":      args.base_url,
        "hw_interval_s": args.hw_interval,
        "agents":        cfg["agents"],
    }
    with open(session_dir / "session_config.json", "w") as f:
        json.dump(session_config_out, f, indent=2)

    # ── Launch agents with staggered offsets ──────────────────────────────────
    trace_py  = REPO_ROOT / "profiling" / "trace_run.py"
    procs     = []
    prev_off  = 0.0

    # ── ONE shared L3+L4 sampler for the whole session (agents skip their own) ──
    # This is the fix for the 2026-08-07 artifact: N per-agent /metrics scrapers
    # flooded SGLang's HTTP frontend and added ~34s to every call. One shared
    # scraper = single-agent load, at full 50ms granularity.
    shared_hw = _tr.HardwareSampler(interval=args.hw_interval)
    shared_sg = _tr.SGLangScraper(base_url=args.base_url, interval=args.hw_interval)
    shared_hw.start(); shared_sg.start()
    sampler_t0_wall = _tr.T0_WALL   # wall time at the sampler's t=0 origin
    print(f"[concurrent] shared L3+L4 sampler @ {args.hw_interval*1000:.0f}ms "
          f"(hw_source={shared_hw.hw_source}) — agents skip their own to avoid /metrics self-load")

    t_session_start = time.time()

    for agent_cfg in cfg["agents"]:
        offset = float(agent_cfg.get("arrival_offset_s", 0))
        delay  = offset - prev_off
        if delay > 0:
            print(f"[concurrent] waiting {delay:.1f}s before {agent_cfg['agent_id']} ...")
            time.sleep(delay)
        prev_off = offset

        agent_outdir = session_dir / "agents" / agent_cfg["agent_id"]
        agent_outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(trace_py),
            "--task-config",        str(agent_cfg["task_config"]),
            "--output-dir",         str(agent_outdir),
            "--concurrent-session", session_id,
            "--base-url",           args.base_url,
            "--hw-interval",        str(args.hw_interval),
        ]
        numactl_node = agent_cfg.get("numactl_node")
        if numactl_node is not None:
            cmd = ["numactl",
                   f"--cpunodebind={numactl_node}",
                   f"--membind={numactl_node}"] + cmd

        log_path = agent_outdir / "launcher.log"
        log_fh   = open(log_path, "w")
        t_launch = time.time()
        proc     = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

        procs.append({
            "agent_cfg":   agent_cfg,
            "proc":        proc,
            "agent_outdir": agent_outdir,
            "t_launch":    t_launch,
            "log_fh":      log_fh,
            "log_path":    log_path,
            "task_id":     _get_task_id(agent_cfg),
        })
        print(f"[concurrent] launched {agent_cfg['agent_id']}  "
              f"PID={proc.pid}  task={Path(agent_cfg['task_config']).stem}")

    # ── Wait for all agents (each auto-analyzes before exiting) ──────────────
    print(f"\n[concurrent] all {n_agents} agents launched — waiting for completion ...")
    results = []
    for p in procs:
        rc      = p["proc"].wait()
        t_done  = time.time()
        p["log_fh"].close()
        wall_s  = round(t_done - p["t_launch"], 2)
        status  = "OK" if rc == 0 else f"FAILED(rc={rc})"
        print(f"[concurrent] {p['agent_cfg']['agent_id']}  {status}  wall={wall_s}s"
              f"  log={p['log_path']}")
        results.append({**p, "returncode": rc, "wall_s": wall_s})

    t_session_end = time.time()

    # ── Stop the shared sampler; reproject its CSVs into each agent's frame + analyze ──
    shared_hw.stop(); shared_sg.stop()
    shared_hw.join(timeout=2 * args.hw_interval + 1)
    shared_sg.join(timeout=2 * args.hw_interval + 1)
    sess_hw = session_dir / "session_hardware.csv"
    sess_sg = session_dir / "session_sglang_metrics.csv"
    shared_hw.dump(sess_hw); shared_sg.dump(sess_sg)
    print(f"\n[concurrent] shared sampler stopped; reprojecting CSVs + analyzing per agent ...")
    for p in procs:
        td = find_trace_dir(p["agent_outdir"], p["task_id"])
        if td:
            _reproject_and_analyze(td, sess_hw, sess_sg, sampler_t0_wall)
            print(f"  {p['agent_cfg']['agent_id']} → analyzed ({td.name})")
        else:
            print(f"  {p['agent_cfg']['agent_id']} → no trace dir; skipped")

    # ── Collect trace dirs (analysis.json now written by the launcher above) ───
    print(f"\n[concurrent] collecting trace directories ...")
    for r in results:
        r["trace_dir"] = find_trace_dir(r["agent_outdir"], r["task_id"])
        status = str(r["trace_dir"]) if r["trace_dir"] else "NOT FOUND"
        print(f"  {r['agent_cfg']['agent_id']} → {status}")

    # ── Build session_summary.json ────────────────────────────────────────────
    print(f"\n[concurrent] building session_summary.json ...")
    agent_summaries       = []
    agent_infos_for_overlap = []

    for r in results:
        aid = r["agent_cfg"]["agent_id"]
        metrics = _extract_agent_metrics(r["trace_dir"])
        agent_summaries.append({
            "agent_id":   aid,
            "task_id":    r["task_id"],
            "returncode": r["returncode"],
            "agent_wall_s": r["wall_s"],
            **metrics,
        })

        gen_spans, exe_spans = (load_events_abs(r["trace_dir"])
                                if r["trace_dir"] else ([], []))
        agent_infos_for_overlap.append({"gen_spans": gen_spans, "exe_spans": exe_spans})

    overlap = compute_overlap_analysis(
        agent_infos_for_overlap, t_session_start, t_session_end)

    # ── Collect single-agent baselines ───────────────────────────────────────
    print(f"\n[concurrent] collecting single-agent baselines ...")
    baselines: dict = {}
    for s in agent_summaries:
        task_id = s["task_id"]
        if task_id in baselines:
            continue
        bdir = find_single_agent_baseline(task_id)
        if bdir:
            baselines[task_id] = _extract_agent_metrics(bdir)
            print(f"  {task_id} → {bdir}")
        else:
            baselines[task_id] = {}
            print(f"  {task_id} → no baseline found")

    # ── Build session_summary.json ────────────────────────────────────────────
    session_summary = {
        "session_id":       session_id,
        "session_name":     cfg.get("session_name", ""),
        "description":      cfg.get("description", ""),
        "n_agents":         n_agents,
        "session_wall_s":   round(t_session_end - t_session_start, 3),
        "agents":           agent_summaries,
        "overlap_analysis": overlap,
        "baselines":        {tid: {k: v for k, v in b.items() if k != "error"}
                             for tid, b in baselines.items()},
    }
    summary_path = session_dir / "session_summary.json"
    with open(summary_path, "w") as f:
        json.dump(session_summary, f, indent=2)

    # ── Build comparison report ───────────────────────────────────────────────
    comparison_report = build_comparison_report(
        cfg.get("session_name", session_id), agent_summaries, baselines)
    comparison_path = session_dir / "session_comparison.txt"
    comparison_path.write_text(comparison_report + "\n")

    # ── Terminal report ───────────────────────────────────────────────────────
    print(f"\n{'═' * 68}")
    print(f"  Session complete: {session_id}")
    print(f"  session_wall_s:   {session_summary['session_wall_s']}")
    print(f"{'─' * 68}")
    for s in agent_summaries:
        bub = s.get("gpu_bubble_weighted_frac")
        bub_str = f"{100*bub:.1f}%" if bub is not None else "n/a"
        tpot = s.get("l3_mean_tpot_ms")
        tpot_str = f"{tpot:.1f}ms" if tpot is not None else "n/a"
        print(f"  {s['agent_id']} ({s['task_id']}): "
              f"bubble={bub_str}  TPOT={tpot_str}  "
              f"execute={s.get('execute_total_s','n/a')}s  "
              f"rc={s['returncode']}")
    print(f"{'─' * 68}")
    ov = overlap
    print(f"  fill_s:           {ov['fill_s']}  ({100*ov['fill_fraction']:.1f}%)")
    print(f"  gpu_collision_s:  {ov['gpu_collision_s']}  "
          f"({100*ov['gpu_collision_fraction']:.1f}%)")
    print(f"  cpu_collision_s:  {ov['cpu_collision_s']}  "
          f"({100*ov['cpu_collision_fraction']:.1f}%)")
    print(f"{'═' * 68}")
    print(f"\n{comparison_report}\n")
    print(f"  Results:    {session_dir}")
    print(f"  Summary:    {summary_path}")
    print(f"  Comparison: {comparison_path}")


if __name__ == "__main__":
    main()
