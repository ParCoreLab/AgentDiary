#!/usr/bin/env python3
"""
Stage-2 raw concurrent-load benchmark — NO Biomni, straight at the SGLang server.

Purpose: isolate the SERVER's true concurrent serving behavior from all agent-framework
overhead. We fire N identical/​distinct completion requests concurrently, each streaming,
and measure CLIENT-side TTFT + TPOT + throughput, while polling num_running_reqs from
/metrics. Sweeping concurrency x prompt-size x {shared,distinct} prefix localizes the cause
of the 34 s TTFT we saw with 10 Biomni agents:

  - if short-prompt concurrency is fine but LONG-prompt TTFT explodes  -> big-context PREFILL (H3)
  - if DISTINCT explodes but SHARED (cache-hit) is fine                -> prefix-cache thrash
  - if even short/distinct caps at ~2 running & TTFT climbs            -> SGLang scheduler/config (H4)
  - if raw load is clean at N=10                                       -> the loss is Biomni-specific

Run against an already-running server on the SAME node (see scripts/mn5_bench.sh).

    python profiling/bench_concurrent.py --model RyanLi0802/Biomni-R0-Preview
"""
import argparse, time, threading, statistics as st, re, json
import multiprocessing as mp
import requests
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:30000"
UNIT = "In this task you will analyze biomedical data step by step. "  # ~11 tokens


def make_prompt(approx_tokens, tag=""):
    reps = max(int(approx_tokens / 11), 1)
    return (tag + " " if tag else "") + (UNIT * reps)


def running_reqs():
    try:
        t = requests.get(BASE + "/metrics", timeout=5).text
        m = re.search(r'sglang:num_running_reqs\S*\s+([0-9.]+)', t)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def one_request(prompt, max_tokens, model):
    t0 = time.perf_counter(); ttft = None; n = 0
    try:
        r = requests.post(BASE + "/v1/completions",
                          json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                                "temperature": 0, "stream": True},
                          stream=True, timeout=1800)
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
    except Exception as e:
        return (None, None, 0, str(e))
    return (ttft, time.perf_counter() - t0, n, None)


def run_level(n_conc, prompt_len, shared, max_tokens, model):
    if shared:
        prompts = [make_prompt(prompt_len, "")] * n_conc
    else:
        prompts = [make_prompt(prompt_len, f"req{i}") for i in range(n_conc)]
    peak = [0.0]; stop = threading.Event()
    def poll():
        while not stop.is_set():
            v = running_reqs()
            if v is not None:
                peak[0] = max(peak[0], v)
            time.sleep(0.2)
    th = threading.Thread(target=poll, daemon=True); th.start()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_conc) as ex:
        res = list(ex.map(lambda p: one_request(p, max_tokens, model), prompts))
    wall = time.perf_counter() - t0
    stop.set(); th.join(timeout=1)
    ok = [r for r in res if r[0] is not None]
    ttfts = [r[0] for r in ok]
    tpots = [(tot - tt) / max(nt - 1, 1) * 1000 for tt, tot, nt, _ in ok if tt and nt > 1]
    thru = sum(r[2] for r in ok) / wall if wall > 0 else 0
    return dict(n=n_conc, prompt_len=prompt_len, shared=shared, ok=len(ok), fail=len(res) - len(ok),
                ttft_mean=st.mean(ttfts) if ttfts else None,
                ttft_med=st.median(ttfts) if ttfts else None,
                ttft_max=max(ttfts) if ttfts else None,
                tpot_mean=st.mean(tpots) if tpots else None,
                thru_tok_s=thru, peak_running=peak[0], wall_s=wall)


def _bg_worker(prompt, max_tokens, model, stop):
    """A sustained long-running request; streams and discards output until done or stop."""
    try:
        r = requests.post(BASE + "/v1/completions",
                          json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                                "temperature": 0, "stream": True}, stream=True, timeout=1800)
        for line in r.iter_lines():
            if stop.is_set():
                r.close(); break
    except Exception:
        pass


def run_probe(n_bg, model):
    """Fill the server with n_bg LONG-running requests (sustained decode), let them settle into
    steady-state decode, then fire fresh probe requests and measure THEIR TTFT. Mimics 'a new agent
    turn arrives while N other agents are mid-generation'. Isolates whether sustained concurrent
    DECODE (not prefill) is what inflates first-token latency."""
    stop = threading.Event(); bg = []
    for i in range(n_bg):
        t = threading.Thread(target=_bg_worker, args=(make_prompt(24000, f"bg{i}"), 4000, model, stop), daemon=True)
        t.start(); bg.append(t); time.sleep(0.2)
    t0 = time.perf_counter(); seen = 0.0
    while time.perf_counter() - t0 < 30:          # wait until backgrounds are actually decoding
        rr = running_reqs()
        if rr is not None:
            seen = max(seen, rr)
            if rr >= max(n_bg * 0.8, 0.5):
                break
        time.sleep(0.5)
    time.sleep(4)                                  # settle into steady-state decode
    run_during = running_reqs()
    probe = []
    for k in range(3):                             # a few fresh probes for a stable number
        ttft, tot, ntok, err = one_request(make_prompt(24000, f"probe{n_bg}_{k}_{time.time()}"), 200, model)
        if ttft is not None:
            probe.append(ttft)
        time.sleep(1)
    stop.set(); time.sleep(2)
    return dict(n_bg=n_bg, running_during=run_during, seen_running=seen,
                probe_ttft_mean=st.mean(probe) if probe else None,
                probe_ttft_max=max(probe) if probe else None, probe_ttft=probe)


def _cpu_burn():
    """Peg one core with a pure-Python busy loop (proxy for an agent process / profiler CPU load)."""
    x = 1.0001
    while True:
        x = x * 1.0000001
        if x > 1e6:
            x = 1.0001


def _metrics_poll(stop):
    """Hammer the server's /metrics endpoint every 50 ms — mimics ONE 4-layer profiler's L3 scraper."""
    while not stop.is_set():
        try:
            requests.get(BASE + "/metrics", timeout=3)
        except Exception:
            pass
        time.sleep(0.05)


def run_cpuload(model):
    """3b: hold the GPU idle, but load the NODE's CPU the way 10 agents + 10 profilers do —
    n_burn busy cores + n_poll /metrics scrapers — then fire fresh probes and measure TTFT.
    If probe TTFT climbs from ~1.4s toward tens of seconds, the 37s is CPU contention starving
    SGLang's CPU-side loop (tokenizer/scheduler/detokenizer/HTTP), NOT the GPU."""
    conditions = [("baseline", 0, 0), ("burn40", 40, 0), ("burn70", 70, 0),
                  ("poll10", 0, 10), ("poll20", 0, 20), ("burn40+poll10", 40, 10)]
    rows = []
    print(f"\n=== 3b: probe TTFT under NODE CPU load (GPU idle) ===")
    print(f"{'condition':>16} {'burn':>5} {'poll':>5} {'probe_TTFT_mean':>16} {'probe_TTFT_max':>15}  {'metrics_ms':>10}")
    print("-" * 78)
    for name, n_burn, n_poll in conditions:
        burners = [mp.Process(target=_cpu_burn, daemon=True) for _ in range(n_burn)]
        for p in burners:
            p.start()
        stop = threading.Event()
        pollers = [threading.Thread(target=_metrics_poll, args=(stop,), daemon=True) for _ in range(n_poll)]
        for t in pollers:
            t.start()
        time.sleep(4)  # let the load ramp
        # how long does a /metrics call itself take under this load? (server HTTP responsiveness)
        t0 = time.perf_counter()
        try:
            requests.get(BASE + "/metrics", timeout=10); metrics_ms = (time.perf_counter() - t0) * 1000
        except Exception:
            metrics_ms = None
        probe = []
        for k in range(3):
            ttft, _, _, _ = one_request(make_prompt(24000, f"cpu{name}_{k}_{time.time()}"), 200, model)
            if ttft is not None:
                probe.append(ttft)
            time.sleep(1)
        stop.set()
        for p in burners:
            p.terminate()
        time.sleep(2)
        f = lambda x: (f"{x:.1f}" if x is not None else "n/a")
        rows.append(dict(cond=name, n_burn=n_burn, n_poll=n_poll,
                         probe_ttft_mean=st.mean(probe) if probe else None,
                         probe_ttft_max=max(probe) if probe else None,
                         metrics_ms=metrics_ms, probe=probe))
        print(f"{name:>16} {n_burn:>5} {n_poll:>5} {f(st.mean(probe) if probe else None):>16} "
              f"{f(max(probe) if probe else None):>15}  {f(metrics_ms):>10}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RyanLi0802/Biomni-R0-Preview")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--mode", choices=["sweep", "probe", "cpuload"], default="sweep")
    ap.add_argument("--out", default="results_bench/bench_concurrent.json")
    args = ap.parse_args()

    # warmup (first request pays CUDA-graph / compile costs)
    print("warmup ..."); one_request(make_prompt(512, "warm"), 8, args.model)

    if args.mode == "probe":
        import os
        print("\n=== 3a: probe-under-sustained-decode (does a new request stall behind ongoing long decodes?) ===")
        print(f"{'n_bg':>5} {'running_during':>14} {'probe_TTFT_mean':>16} {'probe_TTFT_max':>15}")
        print("-" * 54)
        rows = []
        for n_bg in [0, 1, 2, 4, 8]:
            r = run_probe(n_bg, args.model); rows.append(r)
            f = lambda x: (f"{x:.1f}" if x is not None else "n/a")
            print(f"{n_bg:>5} {f(r['running_during']):>14} {f(r['probe_ttft_mean']):>16} {f(r['probe_ttft_max']):>15}")
        out = args.out.replace(".json", "_probe.json")
        os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(rows, open(out, "w"), indent=2)
        print(f"\nwrote {out}")
        print("READ: probe TTFT climbing with n_bg -> sustained concurrent DECODE inflates new-request first-token (suspect #1).")
        print("      probe TTFT ~1s even at n_bg=8 -> decode load is NOT the cause; go to CPU-contention test (3b).")
        return

    if args.mode == "cpuload":
        import os
        rows = run_cpuload(args.model)
        out = args.out.replace(".json", "_cpuload.json")
        os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(rows, open(out, "w"), indent=2)
        print(f"\nwrote {out}")
        print("READ: probe TTFT climbing with burn/poll -> the 37s is NODE CPU contention starving SGLang's CPU-side loop.")
        print("      probe TTFT flat ~1.4s under all load -> CPU is NOT it either; the loss is in the agent/client request path.")
        return

    # (prompt_len, shared, [concurrency levels])
    plan = [
        (512,   False, [1, 2, 4, 8, 10, 16]),   # pure concurrency, tiny prefill
        (24000, False, [1, 2, 4, 8, 10]),        # big-context prefill, no cache (worst case)
        (24000, True,  [1, 2, 4, 8, 10]),        # big-context, shared prefix -> prefix cache should hit
    ]
    rows = []
    hdr = f"{'promptlen':>9} {'shared':>6} {'N':>3} {'ok':>3} {'TTFT_mean':>9} {'TTFT_med':>8} {'TTFT_max':>8} {'TPOT_ms':>7} {'tok/s':>7} {'peakRun':>7}"
    print("\n" + hdr); print("-" * len(hdr))
    for plen, shared, levels in plan:
        for n in levels:
            r = run_level(n, plen, shared, args.max_tokens, args.model)
            rows.append(r)
            f = lambda x, s="{:.1f}": (s.format(x) if x is not None else "n/a")
            print(f"{plen:>9} {str(shared):>6} {n:>3} {r['ok']:>3} {f(r['ttft_mean']):>9} "
                  f"{f(r['ttft_med']):>8} {f(r['ttft_max']):>8} {f(r['tpot_mean']):>7} "
                  f"{f(r['thru_tok_s']):>7} {r['peak_running']:>7.1f}")
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("READ: if 512/distinct stays flat but 24000 TTFT climbs -> big-context prefill is the cause.")
    print("      if 24000/distinct climbs but 24000/shared stays flat -> it's prefix-cache thrash.")
    print("      if peakRun tracks N -> server batches fine; if peakRun stalls ~2 -> scheduler/config cap.")


if __name__ == "__main__":
    main()
