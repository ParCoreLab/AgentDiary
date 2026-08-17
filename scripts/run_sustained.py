#!/usr/bin/env python3
"""
Sustained-load launcher — MAINTAIN ~target concurrent Biomni agents (closed loop), each drawing a
RANDOM task from a mix, for a launch window, then drain. One shared L3+L4 sampler for the whole
session (no per-agent /metrics self-load). Built to stress BOTH resources at once: ~N concurrent
GPU-serving requests AND ~N concurrent CPU-heavy tool phases — to see whether the CPU finally binds.

Unlike run_concurrent.py (fixed batch with arrival offsets), this keeps relaunching agents whenever
the running count drops below --target, so the average concurrency is held at ~target with random
(jittered) arrivals until --duration elapses; then it waits for the stragglers.

Output (per-agent analyze_trace is skipped — too slow at 100s of agents; the SESSION resource traces
are the deliverable):
  session_hardware.csv / session_sglang_metrics.csv   (shared sampler, whole session)
  session_summary.json                                (per-agent generate/execute totals + config)

    python scripts/run_sustained.py --target 100 --duration 900 \
        --tasks tasks/_deprecated_fake/gsea_permutation.json tasks/_deprecated_fake/random_forest_cv_large.json ...
"""
import argparse, json, time, random, subprocess, sys, datetime as dt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "profiling"))
import trace_run as _tr   # shared sampler classes + T0_MONO/T0_WALL


def phase_totals(evpath):
    gen = exe = 0.0; og = oe = None
    try:
        for ln in open(evpath):
            e = json.loads(ln); ph, ed, t = e.get("phase"), e.get("edge"), e.get("t")
            if t is None: continue
            if ph == "generate":
                if ed == "start": og = t
                elif og is not None: gen += t - og; og = None
            elif ph == "execute":
                if ed == "start": oe = t
                elif oe is not None: exe += t - oe; oe = None
    except Exception:
        pass
    return round(gen, 1), round(exe, 1)


def find_trace_dir(agent_outdir: Path, task_id: str = None):
    # trace_run writes to <outdir>/<task_id_FROM_JSON>/<stamp>/, and the json's task_id can differ
    # from the config file's stem (e.g. betweenness_2k_network.json -> task_id betweenness_8k_network).
    # So search for the trace dir instead of assuming its name (the old bug undercounted completions).
    if not agent_outdir.is_dir():
        return None
    for td in sorted(agent_outdir.iterdir()):
        if td.is_dir():
            subs = sorted(d for d in td.iterdir() if d.is_dir())
            if subs:
                return subs[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, required=True, help="concurrent agents to maintain")
    ap.add_argument("--tasks", nargs="+", required=True, help="task-config files to draw from at random")
    ap.add_argument("--duration", type=float, default=900, help="launch window seconds (then drain)")
    ap.add_argument("--max-agents", type=int, default=2000, help="hard safety cap on total launched")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--results-root", default="results_multi")
    ap.add_argument("--hw-interval", type=float, default=0.05)
    ap.add_argument("--jitter", type=float, default=1.0, help="max random seconds between launches")
    args = ap.parse_args()

    session_id = "session_" + dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")
    session_dir = Path(args.results_root) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    task_ids = [Path(t).stem for t in args.tasks]
    print(f"[sustained] {session_id}\n[sustained] target={args.target}  tasks={task_ids}  "
          f"duration={args.duration}s  hw_interval={args.hw_interval}s", flush=True)

    shared_hw = _tr.HardwareSampler(interval=args.hw_interval)
    shared_sg = _tr.SGLangScraper(base_url=args.base_url, interval=args.hw_interval)
    shared_hw.start(); shared_sg.start()
    sampler_t0_wall = _tr.T0_WALL
    trace_py = REPO_ROOT / "profiling" / "trace_run.py"

    procs = []          # {proc, outdir, task, task_id, launch_wall, log}
    launched = 0
    random.seed(0)
    t0 = time.time()

    def running():
        return [p for p in procs if p["proc"].poll() is None]

    last_status = 0
    while (time.time() - t0 < args.duration) or running():
        elapsed = time.time() - t0
        # launch to fill target, only during the launch window
        while (elapsed < args.duration) and (len(running()) < args.target) and (launched < args.max_agents):
            task = random.choice(args.tasks); tid = Path(task).stem
            aid = f"agent_{launched:04d}_{tid}"
            outdir = session_dir / "agents" / aid; outdir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(trace_py), "--task-config", str(task), "--output-dir", str(outdir),
                   "--concurrent-session", session_id, "--base-url", args.base_url,
                   "--hw-interval", str(args.hw_interval)]
            lf = open(outdir / "launcher.log", "w")
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            procs.append(dict(proc=p, outdir=outdir, task=task, task_id=tid, launch=time.time(), log=lf))
            launched += 1
            time.sleep(random.uniform(0.1, args.jitter))
            elapsed = time.time() - t0
        if time.time() - last_status >= 15:
            phase = "LAUNCHING" if elapsed < args.duration else "DRAINING"
            print(f"[sustained] t={elapsed:6.0f}s {phase:9s} running={len(running()):3d} launched={launched} "
                  f"done={launched - len(running())}", flush=True)
            last_status = time.time()
        time.sleep(1.0)

    session_wall = time.time() - t0
    for p in procs:
        p["log"].close()
    shared_hw.stop(); shared_sg.stop()
    shared_hw.join(timeout=2 * args.hw_interval + 1); shared_sg.join(timeout=2 * args.hw_interval + 1)
    shared_hw.dump(session_dir / "session_hardware.csv")
    shared_sg.dump(session_dir / "session_sglang_metrics.csv")

    # per-agent phase summary (from events.jsonl — no per-agent analyze_trace at this scale)
    agents = []
    for p in procs:
        td = find_trace_dir(p["outdir"], p["task_id"])
        rc = p["proc"].returncode
        rec = {"agent_id": p["outdir"].name, "task_id": p["task_id"], "returncode": rc}
        if td and (td / "events.jsonl").exists():
            g, e = phase_totals(td / "events.jsonl")
            rec["generate_total_s"] = g; rec["execute_total_s"] = e
            try:
                rec["error"] = json.loads((td / "meta.json").read_text()).get("error")
            except Exception:
                rec["error"] = None
        agents.append(rec)
    json.dump({"session_id": session_id, "mode": "sustained", "target": args.target,
               "tasks": task_ids, "duration_s": args.duration, "launched": launched,
               "session_wall_s": round(session_wall, 1), "sampler_t0_wall_unix": sampler_t0_wall,
               "agents": agents},
              open(session_dir / "session_summary.json", "w"), indent=2)
    ok = sum(1 for a in agents if a.get("returncode") == 0 and (a.get("execute_total_s") or 0) > 5)
    print(f"[sustained] done: launched {launched}, {ok} completed with real execute, session {session_wall:.0f}s")
    print(f"[sustained] -> {session_dir}/session_hardware.csv + session_sglang_metrics.csv")


if __name__ == "__main__":
    main()
