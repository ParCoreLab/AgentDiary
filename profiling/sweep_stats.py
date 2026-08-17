#!/usr/bin/env python3
"""
Distribution summary of a sweep (the deliverable of mn5_sweep.sh: N repeats of a task).
Reads an aggregate.csv (one row per run, from aggregate_traces.py), groups by task_id, and reports
mean / std / CV% / p10 / p50 / p90 / min / max for the key timing + serving-quality metrics, plus the
completion rate. This is the "like the other tasks" distribution table.

  python profiling/sweep_stats.py [results/aggregate.csv] [--tasks depmap_nmf_light depmap_nmf_heavy ...]

NOTE: per-phase I/O (disk_read during the load vs compute phases) is NOT in aggregate.csv yet -- for the
BindingDB I/O characterization, analyze the raw hardware.csv (disk_read_mbps / ram_used_gb) per run; the
task also self-reports its load time. A later analyze_trace.py enhancement will fold I/O into aggregate.csv.
"""
import sys, csv, statistics as st
from collections import defaultdict

METRICS = [
    ("wall_time_s", "wall (s)"), ("generate_total_s", "generate (s)"), ("execute_total_s", "execute (s)"),
    ("gpu_bubble_weighted_frac", "GPU bubble"), ("l2_total_completion_tokens", "compl tokens"),
    ("l2_n_llm_calls", "n_llm_calls"), ("l3_mean_tpot_ms", "TPOT (ms)"), ("l3_mean_ttft_s", "TTFT (s)"),
]


def pctile(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def num(x):
    try:
        if x in (None, "", "nan", "None"):
            return None
        return float(x)
    except (ValueError, TypeError):
        return None


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = pos[0] if pos else "results/aggregate.csv"
    tasks_filter = None
    if "--tasks" in sys.argv:
        i = sys.argv.index("--tasks")
        tasks_filter = set(a for a in sys.argv[i + 1:] if not a.startswith("--"))

    rows = list(csv.DictReader(open(path)))
    by = defaultdict(list)
    for r in rows:
        by[r.get("task_id", "?")].append(r)

    for task in sorted(by):
        if tasks_filter and task not in tasks_filter:
            continue
        runs = by[task]
        ok = [r for r in runs if not (r.get("error") and r["error"] not in ("", "None"))]
        print(f"\n=== {task}   n={len(runs)}  completed={len(ok)} ({100*len(ok)//max(1,len(runs))}%)  "
              f"failed={len(runs)-len(ok)} ===")
        print(f"  {'metric':14s}{'mean':>10}{'std':>9}{'CV%':>7}{'p10':>10}{'p50':>10}{'p90':>10}{'min':>10}{'max':>10}")
        for key, label in METRICS:
            vals = [num(r.get(key)) for r in ok]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            m = st.mean(vals)
            sd = st.pstdev(vals) if len(vals) > 1 else 0.0
            cv = 100 * sd / m if m else 0.0
            print(f"  {label:14s}{m:10.2f}{sd:9.2f}{cv:7.1f}{pctile(vals,.1):10.2f}"
                  f"{pctile(vals,.5):10.2f}{pctile(vals,.9):10.2f}{min(vals):10.2f}{max(vals):10.2f}")


if __name__ == "__main__":
    main()
