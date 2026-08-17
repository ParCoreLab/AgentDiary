#!/usr/bin/env python3
"""
PIPELINE — generate the unified 6-panel multi-run figure (per-agent Gantt + overlap strip + GPU SM/DRAM/occ
+ KV-cache/#running + CPU + TTFT/TPOT) for ANY run_sustained / CPU-throttle session, via plot_sustained.py.
One command regenerates figures for a whole class of runs. Saves to documentation/figs/fig_concurrent_<label>.png
AND drops a copy inside the session dir (fig_concurrent_sustained.png).

  python scripts/make_multi_figs.py throttle        # all results_mn5_multi/throttle_*c/session_* runs
  python scripts/make_multi_figs.py sustained       # all top-level results_mn5_multi/session_* runs
  python scripts/make_multi_figs.py all             # both
  python scripts/make_multi_figs.py <session_dir> ...   # explicit sessions

label = "throttle_<N>c" (from path) or "sustained<target>" (from session_summary.json). Idempotent — safe to
re-run after new sessions land. Requires per-agent events.jsonl+meta.json + session_*.csv/json in each session.
"""
import sys, glob, json, re, shutil, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOT = REPO / "profiling" / "plot_sustained.py"
FIGS = REPO / "documentation" / "figs"
ROOT = REPO / "results_mn5_multi"


def label_for(sdir: Path):
    m = re.search(r"throttle_(\d+)c", str(sdir))
    if m:
        return f"throttle_{m.group(1)}c"
    try:
        t = json.load(open(sdir / "session_summary.json")).get("target")
        return f"sustained{t}"
    except Exception:
        return sdir.name


def expand(arg):
    if arg == "throttle":
        return sorted(glob.glob(str(ROOT / "throttle_*c" / "session_*")))
    if arg == "sustained":
        return sorted(glob.glob(str(ROOT / "session_*")))
    if arg == "all":
        return sorted(glob.glob(str(ROOT / "session_*"))) + sorted(glob.glob(str(ROOT / "throttle_*c" / "session_*")))
    return [arg]


def main():
    args = sys.argv[1:] or ["throttle"]
    sdirs = []
    for a in args:
        sdirs += expand(a)
    if not sdirs:
        print("no sessions matched", args); return
    FIGS.mkdir(parents=True, exist_ok=True)
    ok = 0
    for s in sdirs:
        sdir = Path(s)
        if not (sdir / "session_summary.json").exists():
            print("  skip (no session_summary.json):", sdir); continue
        label = label_for(sdir)
        out = FIGS / f"fig_concurrent_{label}.png"
        cmd = [sys.executable, str(PLOT), "--session", str(sdir), "--ncpu", "160", "--out", str(out)]
        if label.startswith("throttle_"):
            n = label.split("_")[1].rstrip("c")
            cmd += ["--title", f"Biomni CPU-throttle — 40 agents on {n} CPU cores (server separate)  ·  {label}"]
        print(f"[fig] {label:16s} <- {sdir.relative_to(REPO) if str(sdir).startswith(str(REPO)) else sdir}")
        r = subprocess.run(cmd)
        if r.returncode == 0:
            shutil.copy(out, sdir / "fig_concurrent_sustained.png")
            ok += 1
        else:
            print("   !! plot_sustained.py failed for", sdir)
    print(f"done — {ok}/{len(sdirs)} figures in {FIGS}/ (+ a copy in each session dir)")


if __name__ == "__main__":
    main()
