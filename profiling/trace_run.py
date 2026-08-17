#!/usr/bin/env python3
"""
trace_run.py  -- four-layer structural tracer for one Biomni agent program.

Captures:
  Layer 1 — Agent-loop phase boundaries (generate / execute start/end),
             written to events.jsonl on the shared monotonic clock.
  Layer 2 — Per-LLM-call HTTP view: end-to-end latency, prompt tokens,
             completion tokens.  Embedded in events.jsonl as extra fields on
             generate:end events.  No additional HTTP round-trip.
  Layer 3 — SGLang engine metrics scraped from the Prometheus /metrics
             endpoint at ~100 ms intervals: TTFT/TPOT histograms, queue depth,
             KV-cache usage, forward-execution time split by prefill vs decode.
             Written to sglang_metrics.csv on the same monotonic clock.
  Layer 4 — Hardware telemetry at 20–50 ms intervals.  Primary: DCGM
             (SM-Activity, SM-Occupancy, DRAM-Activity, power, clock,
             framebuffer free/used).  Fallback: NVML (SM-util, memory-BW-util,
             same capacity/power/clock fields).  Written to hardware.csv.
             CPU (per-core, mean, max, hot-cores) via psutil.

All timestamps use time.perf_counter() minus T0_MONO so every row in every
file joins on the same axis.  One time.time() anchor in meta.json relates
the trace to real time and SGLang server logs.

Usage:
    python profiling/trace_run.py --task-config tasks/scrna_pbmc3k.json

    # override sampling interval (seconds):
    python profiling/trace_run.py --task-config tasks/umap_large_synthetic.json \\
        --hw-interval 0.02

Output directory: results/{task_id}/{YYYY-MM-DD_HH-MM-SS-ffffff}/
Files written:
    events.jsonl         Layer-1 + Layer-2 event stream
    hardware.csv         Layer-4 GPU (DCGM or NVML) + CPU samples
    sglang_metrics.csv   Layer-3 SGLang Prometheus scrapes
    summary.json         quick per-run summary (Layer-1)
    meta.json            config + clock anchors + hw_source flag
    agent_output.txt     agent final answer (sanity check)
    trace.log            full stdout + stderr

Analyze with:
    python profiling/analyze_trace.py results/{task_id}/{timestamp}/
"""

import os

# du04 has 256 cores but the pre-built OpenBLAS supports max 128 threads.
# Concurrent callers (agent thread + numba threads) each try to spawn N workers
# → total across callers exceeds 128 → segfault. Force-set to 8 (not setdefault)
# so even many concurrent callers stay well under the 128 limit.
# Must happen before any numpy/scipy/sklearn/umap import in this process tree.
for _ev in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_ev] = "8"

import argparse
import csv
import datetime as dt
import itertools
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  Log tee: mirror every print() / write to sys.stdout/stderr to trace.log.
# ─────────────────────────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, stream, log_file):
        self._orig = stream
        self._log  = open(log_file, "a", buffering=1, encoding="utf-8", errors="replace")
    def write(self, data):
        self._orig.write(data); self._orig.flush()
        self._log.write(data)
    def flush(self):
        self._orig.flush(); self._log.flush()
    def fileno(self):
        return self._orig.fileno()
    def close(self):
        self._log.close()
    @property
    def orig(self):
        return self._orig

def _start_logging(log_path: Path):
    tee_out = _Tee(sys.stdout, log_path)
    tee_err = _Tee(sys.stderr, log_path)
    sys.stdout = tee_out; sys.stderr = tee_err
    return tee_out, tee_err

def _stop_logging(tee_out, tee_err):
    sys.stdout = tee_out.orig; sys.stderr = tee_err.orig
    tee_out.close(); tee_err.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Shared clock: every event and every sample uses perf_counter() (monotonic).
# ─────────────────────────────────────────────────────────────────────────────
T0_MONO = time.perf_counter()
T0_WALL = time.time()

def now() -> float:
    return time.perf_counter() - T0_MONO


# ─────────────────────────────────────────────────────────────────────────────
#  Event log: thread-safe append of phase-boundary records (Layers 1 + 2).
# ─────────────────────────────────────────────────────────────────────────────
class EventLog:
    def __init__(self):
        self._events = []
        self._lock   = threading.Lock()

    def add(self, phase: str, edge: str, **extra):
        rec = {"t": now(), "phase": phase, "edge": edge}
        rec.update(extra)
        with self._lock:
            self._events.append(rec)

    def dump(self, path: Path):
        with self._lock, open(path, "w") as f:
            for rec in self._events:
                f.write(json.dumps(rec) + "\n")

    @property
    def events(self):
        with self._lock:
            return list(self._events)

EVENTS = EventLog()


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 2 — LLM call token capture.
#  Extends the existing generate-phase class-level hook to also extract token
#  counts and end-to-end HTTP latency from the LangChain AIMessage response.
# ─────────────────────────────────────────────────────────────────────────────
_llm_call_counter = itertools.count()
# Optional TIGHT cap on total LLM calls per run. A runaway agent that never emits a runnable
# code block loops the generate node (re-prompted "you must include a tag"), so call count
# climbs without the task ever completing. Capping calls aborts it after "a few loops" and
# lets trace_run.py record the failure (agent_completed=False) instead of wasting minutes.
# Off by default (multi-turn tasks make many legit calls); set per single-turn-task sweeps, e.g.
# MAX_LLM_CALLS=8 for gsea (which needs 3).
_MAX_LLM_CALLS = int(os.environ["MAX_LLM_CALLS"]) if os.environ.get("MAX_LLM_CALLS") else None
# Optional WALL-CLOCK cap per run, checked at each LLM call so a runaway is aborted GRACEFULLY
# and RECORDED (agent_completed=False) — unlike an external `timeout` SIGKILL, which writes no
# trace at all and makes the failure vanish from aggregate.csv. Set via env (seconds).
_MAX_WALL_SECONDS = float(os.environ["MAX_WALL_SECONDS"]) if os.environ.get("MAX_WALL_SECONDS") else None
_trace_wall_start = None

# GENERATE-RUNAWAY guard (evidence-based, 2026-08-08). A degenerate model can emit max-length
# completions forever WITHOUT ever producing a parseable tool call (observed at 40×/100×: 8192-token
# completions ×9, 0 tools executed, ~6% of runs). MAX_LLM_CALLS can't catch it — the runaway needs
# only ~9 giant calls, never reaching 20 — so MAX_WALL is the only backstop and wastes ~30min of GPU.
# This guard aborts after N CONSECUTIVE max-length completions while NO tool has EVER executed — the
# exact runaway signature. Healthy agents never exceeded 1 consecutive max-length call, so this is
# provably free of false-positives. Off unless MAX_CONSEC_MAXLEN is set (like the other caps).
_MAX_CONSEC_MAXLEN = int(os.environ["MAX_CONSEC_MAXLEN"]) if os.environ.get("MAX_CONSEC_MAXLEN") else None
_MAXLEN_TOKENS = int(os.environ.get("MAXLEN_TOKENS", "8000"))   # completion at/above this == truncated (cap 8192)
_consec_maxlen = 0
_any_execute_done = False


def _extract_tokens(result):
    """
    Extract (prompt_tokens, completion_tokens, total_tokens) from a LangChain
    AIMessage.  Returns (None, None, None) if unavailable.

    LangChain exposes token counts via two paths:
      • result.usage_metadata  — newer LangChain (≥0.2.x)
        keys: input_tokens, output_tokens, total_tokens
      • result.response_metadata["token_usage"]  — older LangChain / OpenAI compat
        keys: prompt_tokens, completion_tokens, total_tokens
    """
    if result is None:
        return None, None, None
    try:
        um = getattr(result, "usage_metadata", None)
        if um:
            return (um.get("input_tokens"),
                    um.get("output_tokens"),
                    um.get("total_tokens"))
    except Exception:
        pass
    try:
        rm = getattr(result, "response_metadata", None) or {}
        tu = rm.get("token_usage") or {}
        if tu:
            return (tu.get("prompt_tokens"),
                    tu.get("completion_tokens"),
                    tu.get("total_tokens"))
    except Exception:
        pass
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Instrumentation: wrap phase functions non-invasively (no Biomni edits).
# ─────────────────────────────────────────────────────────────────────────────
def _wrap_execute():
    import biomni.tool.support_tools as st
    for fname, lang in [
        ("run_python_repl", "python"),
        ("run_r_code",      "r"),
        ("run_bash_script", "bash"),
    ]:
        orig = getattr(st, fname, None)
        if orig is None:
            print(f"[wrap] {fname} not found; skipping.")
            continue

        def make(orig, lang, fname):
            def wrapped(*args, **kwargs):
                global _any_execute_done
                _any_execute_done = True   # a real tool ran → this agent is productive; disable runaway guard
                EVENTS.add("execute", "start", lang=lang, fn=fname)
                try:
                    return orig(*args, **kwargs)
                finally:
                    EVENTS.add("execute", "end", lang=lang, fn=fname)
            return wrapped

        setattr(st, fname, make(orig, lang, fname))
        print(f"[wrap] execute hook installed on {fname}")


def _wrap_generate(llm_obj):
    """
    Wrap LLM.invoke() at the class level (Pydantic models reject instance-attr
    assignment).  This single wrap serves both Layer-1 (phase boundary) and
    Layer-2 (token counts + per-call latency).

    Layer-2 extras emitted on generate:end events:
      seq              — monotonic call sequence number (matches start ↔ end)
      latency_s        — end-to-end wall time from before invoke() to after
      prompt_tokens    — prompt/input token count (from response metadata)
      completion_tokens— generated token count
      total_tokens     — prompt + completion
    """
    cls = type(llm_obj)
    if getattr(cls, "_biomni_traced", False):
        print(f"[wrap] generate hook already installed on {cls.__name__}.invoke")
        return llm_obj
    if not hasattr(cls, "invoke"):
        print(f"[wrap] {cls.__name__} has no .invoke(); generate phase NOT hooked.")
        return llm_obj

    orig = cls.invoke

    def wrapped(self, *args, **kwargs):
        seq = next(_llm_call_counter)
        global _trace_wall_start
        if _trace_wall_start is None:
            _trace_wall_start = now()
        if _MAX_LLM_CALLS is not None and seq >= _MAX_LLM_CALLS:
            EVENTS.add("generate", "end", seq=seq, error=True)
            raise RuntimeError(
                f"MAX_LLM_CALLS={_MAX_LLM_CALLS} exceeded (call #{seq}) — agent looping "
                f"without completing (runaway); aborting so the run is recorded as failed."
            )
        if _MAX_WALL_SECONDS is not None and (now() - _trace_wall_start) > _MAX_WALL_SECONDS:
            EVENTS.add("generate", "end", seq=seq, error=True)
            raise RuntimeError(
                f"MAX_WALL_SECONDS={_MAX_WALL_SECONDS} exceeded "
                f"({now() - _trace_wall_start:.0f}s elapsed) — agent running too long (runaway); "
                f"aborting so the run is recorded as failed."
            )
        t_before = now()
        EVENTS.add("generate", "start", seq=seq)
        try:
            result = orig(self, *args, **kwargs)
            latency_s = round(now() - t_before, 4)
            p_tok, c_tok, tot_tok = _extract_tokens(result)
            EVENTS.add("generate", "end",
                       seq=seq,
                       latency_s=latency_s,
                       prompt_tokens=p_tok,
                       completion_tokens=c_tok,
                       total_tokens=tot_tok)
        except Exception:
            EVENTS.add("generate", "end", seq=seq, error=True)
            raise
        # ── GENERATE-RUNAWAY guard (checked AFTER the call — needs completion_tokens) ──
        global _consec_maxlen
        if c_tok is not None and c_tok >= _MAXLEN_TOKENS:
            _consec_maxlen += 1
        else:
            _consec_maxlen = 0
        if (_MAX_CONSEC_MAXLEN is not None and _consec_maxlen >= _MAX_CONSEC_MAXLEN
                and not _any_execute_done):
            EVENTS.add("generate", "guard_abort", seq=seq, consec_maxlen=_consec_maxlen)
            raise RuntimeError(
                f"GENERATE-RUNAWAY guard: {_consec_maxlen} consecutive max-length "
                f"(>={_MAXLEN_TOKENS}-tok) completions with 0 tools executed — model looping "
                f"without producing a tool call; aborting so the run is recorded as failed."
            )
        return result

    cls.invoke   = wrapped
    cls._biomni_traced = True
    print(f"[wrap] generate hook installed on {cls.__name__}.invoke (Layer-1 + Layer-2)")
    return llm_obj


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 3 — SGLang Prometheus scraper.
#  Polls /metrics at the same interval as hardware sampling (~100 ms).
#  Writes sglang_metrics.csv with a fixed schema of the metrics we care about.
#
#  Prometheus text-format parser strategy:
#    • Skip #HELP / #TYPE / _bucket lines.
#    • For each data line: extract (metric_name, label_dict, value).
#    • Map to csv_key using SGLANG_CAPTURE table; ignore unrecognised metrics.
#  This is intentionally conservative: explicit > implicit for a fixed schema.
# ─────────────────────────────────────────────────────────────────────────────

# (prometheus_metric_name, label_filter | None)  →  csv_column_name
# label_filter is a tuple of (key, value) pairs; None matches any labels.
# Must be tuples (not dicts) so the outer tuple is hashable as a dict key.
_SGLANG_CAPTURE = {
    # ── Scheduler state gauges (instantaneous) ────────────────────────────
    ("sglang:num_running_reqs",    None): "num_running_reqs",
    ("sglang:num_queue_reqs",      None): "num_queue_reqs",
    ("sglang:gen_throughput",      None): "gen_throughput",
    ("sglang:cache_hit_rate",      None): "cache_hit_rate",
    ("sglang:decode_sum_seq_lens", None): "decode_sum_seq_lens",
    # ── KV-cache pool ─────────────────────────────────────────────────────
    ("sglang:token_usage",         None): "token_usage",
    ("sglang:kv_available_tokens", None): "kv_available_tokens",
    ("sglang:kv_evictable_tokens", None): "kv_evictable_tokens",
    ("sglang:kv_used_tokens",      None): "kv_used_tokens",
    ("sglang:num_used_tokens",     None): "num_used_tokens",
    ("sglang:max_total_num_tokens",None): "max_total_num_tokens",
    # ── GPU forward-execution time (cumulative; take delta per span) ──────
    ("sglang:forward_execution_seconds_total", (("category", "extend"),)): "fwd_exec_extend_s",
    ("sglang:forward_execution_seconds_total", (("category", "decode"),)): "fwd_exec_decode_s",
    ("sglang:forward_execution_seconds_total", (("category", "mixed"),)):  "fwd_exec_mixed_s",
    # ── Realtime-token counters (cumulative; take delta) ──────────────────
    ("sglang:realtime_tokens_total", (("mode", "prefill_compute"),)): "tokens_prefill_compute",
    ("sglang:realtime_tokens_total", (("mode", "prefill_cache"),)):   "tokens_prefill_cache",
    ("sglang:realtime_tokens_total", (("mode", "decode"),)):          "tokens_decode",
    # ── Request-level cumulative counters ─────────────────────────────────
    ("sglang:prompt_tokens_total",     None): "prompt_tokens_total",
    ("sglang:generation_tokens_total", None): "generation_tokens_total",
    ("sglang:num_requests_total",      None): "num_requests_total",
    # ── Latency histograms (cumulative _sum + _count) ─────────────────────
    ("sglang:time_to_first_token_seconds_sum",   None): "ttft_sum",
    ("sglang:time_to_first_token_seconds_count", None): "ttft_count",
    ("sglang:inter_token_latency_seconds_sum",   None): "tpot_sum",
    ("sglang:inter_token_latency_seconds_count", None): "tpot_count",
    ("sglang:e2e_request_latency_seconds_sum",   None): "e2e_sum",
    ("sglang:e2e_request_latency_seconds_count", None): "e2e_count",
    ("sglang:queue_time_seconds_sum",            None): "queue_time_sum",
    ("sglang:queue_time_seconds_count",          None): "queue_time_count",
    # ── Per-stage latency ─────────────────────────────────────────────────
    ("sglang:per_stage_req_latency_seconds_sum",   (("stage", "extend"),)): "stage_extend_sum",
    ("sglang:per_stage_req_latency_seconds_sum",   (("stage", "decode"),)): "stage_decode_sum",
    ("sglang:per_stage_req_latency_seconds_count", (("stage", "extend"),)): "stage_extend_count",
    ("sglang:per_stage_req_latency_seconds_count", (("stage", "decode"),)): "stage_decode_count",
    # ── Engine utilization ────────────────────────────────────────────────
    ("sglang:utilization",  None): "utilization",
    ("sglang:fwd_occupancy",None): "fwd_occupancy",
}

# Fixed fieldnames for sglang_metrics.csv (same order every time).
SGLANG_FIELDNAMES = ["t"] + sorted(set(_SGLANG_CAPTURE.values()))


def _parse_prometheus_line(line: str):
    """
    Parse one Prometheus text-format data line.
    Returns (metric_name, label_dict, value) or None on failure.
    Skips #HELP/#TYPE and _bucket lines.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    try:
        # Split on the last space to isolate value (handles spaces in labels).
        last_sp = line.rfind(" ")
        if last_sp == -1:
            return None
        name_labels = line[:last_sp].strip()
        value_str   = line[last_sp:].strip()
        value = float(value_str)
    except ValueError:
        return None

    if "{" in name_labels:
        brace = name_labels.index("{")
        metric_name = name_labels[:brace]
        label_str   = name_labels[brace + 1:].rstrip("}")
        labels = {}
        for part in label_str.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                labels[k.strip()] = v.strip().strip('"')
    else:
        metric_name = name_labels
        labels = {}

    # Skip bucket lines (keep _sum and _count for histograms).
    if metric_name.endswith("_bucket"):
        return None

    return metric_name, labels, value


def _scrape_to_row(text: str) -> dict:
    """Convert a full Prometheus /metrics response text to a flat csv-column dict."""
    row = {}
    for line in text.splitlines():
        parsed = _parse_prometheus_line(line)
        if parsed is None:
            continue
        metric_name, labels, value = parsed
        for (cap_name, cap_filter), csv_key in _SGLANG_CAPTURE.items():
            if metric_name != cap_name:
                continue
            if cap_filter is None or all(labels.get(k) == v for k, v in cap_filter):
                row[csv_key] = value
                break
    return row


class SGLangScraper(threading.Thread):
    """
    Layer-3: background thread scraping SGLang's Prometheus /metrics endpoint.

    Interval: same as hardware sampling (default 100 ms) so every Layer-4
    hardware sample has a corresponding Layer-3 row within ≤ 100 ms — close
    enough to join on the shared monotonic clock without interpolation.

    Output: sglang_metrics.csv with SGLANG_FIELDNAMES columns.
    Cumulative counters (fwd_exec_*_s, ttft_sum, etc.) are stored raw;
    analyze_trace.py computes inter-scrape deltas to get per-span values.
    """

    def __init__(self, base_url: str, interval: float = 0.1):
        super().__init__(daemon=True, name="SGLangScraper")
        self.interval  = interval
        self._base     = base_url.removesuffix("/v1").rstrip("/")
        self._url      = f"{self._base}/metrics"
        self.rows      = []
        self._stop_evt = threading.Event()
        self._ok       = False

        # Probe once at startup.
        try:
            with urllib.request.urlopen(self._url, timeout=2) as r:
                text = r.read().decode("utf-8")
            self._ok = True
            # Report what we actually found.
            found = set(_scrape_to_row(text).keys())
            print(f"[sglang-scraper] reachable at {self._url}")
            print(f"[sglang-scraper] columns found: {sorted(found)}")
        except Exception as e:
            print(f"[sglang-scraper] not reachable ({e}); sglang_metrics.csv will be empty.")

    def run(self):
        while not self._stop_evt.is_set():
            t = now()
            row = {"t": t}
            if self._ok:
                try:
                    with urllib.request.urlopen(self._url, timeout=0.8) as r:
                        text = r.read().decode("utf-8")
                    row.update(_scrape_to_row(text))
                except Exception:
                    pass  # keep last known structure; missing values → blank
            self.rows.append(row)
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()

    def dump(self, path: Path):
        if not self.rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SGLANG_FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in SGLANG_FIELDNAMES})
        print(f"[sglang-scraper] wrote {len(self.rows)} rows → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 4 — DCGM collector (primary GPU telemetry path).
#
#  Uses DcgmReader from the system DCGM bindings to connect to the running
#  nv-hostengine daemon and request fine-grained per-GPU profiling fields.
#
#  Field semantics (all ratios [0,1] multiplied by 100 → %):
#    1002  SM Activity    — fraction of cycles with ≥1 active warp.  Reported
#          as "100%" by NVML even when decode is fully memory-bandwidth-bound
#          and SMs are stalled waiting for HBM data.  Do not confuse with
#          "GPU is compute-saturated."
#    1003  SM Occupancy   — fraction of max possible warps actually scheduled.
#          Low during decode (memory-bandwidth-bound); high during prefill.
#          NOT available from NVML.  This is the key metric distinguishing
#          prefill (compute-bound, high occupancy) from decode (BW-bound, low).
#    1005  DRAM Activity  — HBM bandwidth utilization.  Near 100% during
#          decode (the true bottleneck); lower during prefill.  Together with
#          SM-Activity and SM-Occupancy, this fully characterises which
#          resource a given generate phase is saturating.
#    155   Board power (W)  — proxy for overall compute intensity.
#    100   SM clock (MHz)   — detect thermal throttling.
#    251   FB Free (MB)     — headroom for a second tenant's KV cache.
#    252   FB Used (MB)     — total HBM committed (weights + KV cache).
# ─────────────────────────────────────────────────────────────────────────────
_DCGM_BINDINGS = os.environ.get(
    "DCGM_BINDINGS_PATH", "/usr/share/datacenter-gpu-manager-4/bindings/python3"
)

_DCGM_FIELD_IDS = {
    "sm_act":       1002,  # SM Activity ratio     → × 100 for %
    "sm_occ":       1003,  # SM Occupancy ratio    → × 100 for %
    "dram_act":     1005,  # DRAM Activity ratio   → × 100 for %
    "power_w":       155,  # Board power in Watts  (DCGM_FI_DEV_BOARD_POWER_WATTS)
    "sm_clock_mhz": 100,   # SM clock in MHz       (DCGM_FI_DEV_SM_CLOCK)
    "mem_free_mb":  251,   # Free framebuffer MB   (DCGM_FI_DEV_FB_FREE)
    "mem_used_mb":  252,   # Used framebuffer MB   (DCGM_FI_DEV_FB_USED)
}


class DcgmCollector:
    """
    Thin wrapper around DcgmReader for on-demand polling.
    Connects to the running nv-hostengine; raises if unavailable (caller
    catches and falls back to NVML).
    """

    def __init__(self, update_freq_us: int = 20_000):
        if _DCGM_BINDINGS not in sys.path:
            sys.path.insert(0, _DCGM_BINDINGS)

        from DcgmReader import DcgmReader  # noqa: F401 — needs path set above

        self._field_ids = list(_DCGM_FIELD_IDS.values())
        # Use a PID-unique group name — a previous run killed with SIGKILL leaves
        # a stale "biomni_profiler" group in nv-hostengine that returns zeros for
        # profiling fields (1002/1003/1005) while non-profiling fields still work.
        self._dr = DcgmReader(
            hostname="localhost",
            fieldIds=self._field_ids,
            updateFrequency=update_freq_us,
            maxKeepAge=300.0,
            fieldGroupName=f"biomni_profiler_{os.getpid()}",
            ignoreBlank=True,
        )
        self._dr.Init()
        # Probe: get GPU IDs and verify data flows.
        test = self._dr.GetLatestGpuValuesAsFieldIdDict()
        if not test:
            raise RuntimeError("DCGM connected but returned no GPU data.")
        self._gpu_ids = sorted(test.keys())
        self.n_gpu = len(self._gpu_ids)
        # Warn if profiling fields return zero — symptom of stale group or
        # another client holding the exclusive profiling resource.
        first_gpu = self._gpu_ids[0]
        prof_vals = [test[first_gpu].get(fid, 0) for fid in (1002, 1003, 1005)]
        if all(v == 0 for v in prof_vals):
            print("[dcgm] WARNING: profiling fields (SM-Act/SM-Occ/DRAM-Act) all zero at init — "
                  "another client may hold the profiling resource, or nv-hostengine needs restart.")
        print(f"[dcgm] ready: {self.n_gpu} GPUs, update_freq={update_freq_us} µs")

    def read(self) -> dict:
        """
        Returns {gpu_idx (0-based): {field_name: value}}.
        Ratio fields (sm_act, sm_occ, dram_act): multiplied × 100 → percent.
        Returns {} on error.
        """
        try:
            raw = self._dr.GetLatestGpuValuesAsFieldIdDict()
        except Exception:
            return {}
        result = {}
        for idx, gpu_id in enumerate(self._gpu_ids):
            gpu_raw = raw.get(gpu_id, {})
            gpu_data = {}
            for fname, fid in _DCGM_FIELD_IDS.items():
                v = gpu_raw.get(fid)
                if v is None:
                    gpu_data[fname] = None
                elif fid in (1002, 1003, 1005):        # ratio → percent
                    gpu_data[fname] = round(float(v) * 100.0, 2)
                else:
                    gpu_data[fname] = v
            result[idx] = gpu_data
        return result

    def shutdown(self):
        try:
            self._dr.Shutdown()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 4 — Hardware sampler thread (DCGM primary, NVML fallback + CPU).
#
#  Column naming in hardware.csv uses DCGM canonical names regardless of
#  backend so analyze_trace.py needs no branching:
#    gpu{i}_sm_act      SM activity %     (DCGM 1002 or NVML util.gpu)
#    gpu{i}_sm_occ      SM occupancy %    (DCGM 1003 only; NVML → blank)
#    gpu{i}_dram_act    DRAM activity %   (DCGM 1005 or NVML util.memory)
#    gpu{i}_power_w     power W           (DCGM 155 in W; NVML in mW / 1000)
#    gpu{i}_sm_clock_mhz SM clock MHz
#    gpu{i}_mem_free_mb free HBM MB
#    gpu{i}_mem_used_mb used HBM MB
#    hw_source          "dcgm" or "nvml"
#    cpu{i}             per-core util %
#    cpu_mean           mean across all cores
#    cpu_max            hottest single core (catches pinned-core tools)
#    cpu_hot_cores      count of cores > 50%
# ─────────────────────────────────────────────────────────────────────────────
_GPU_SUFFIXES = ("sm_act", "sm_occ", "dram_act", "power_w",
                 "sm_clock_mhz", "mem_free_mb", "mem_used_mb")


def _discover_server_procs(psutil_mod, my_pid: int):
    """
    Find all SGLang server worker processes by scanning /proc cmdlines.

    Returns a list of psutil.Process objects already primed with an initial
    cpu_percent() call (so subsequent calls give valid interval-based readings).

    Raises RuntimeError if no SGLang processes are found — the server MUST be
    running before trace_run.py starts. If the process list looks right in
    `ps aux | grep sglang` but this still raises, the match string in this
    function needs updating to match the actual command-line tokens on this
    SGLang version.
    """
    candidates = []
    for proc in psutil_mod.process_iter(["pid", "name", "cmdline", "status"]):
        try:
            if proc.info["pid"] == my_pid:
                continue
            if proc.info.get("status") in (psutil_mod.STATUS_ZOMBIE,
                                           psutil_mod.STATUS_DEAD):
                continue
            name    = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if "sglang" in name or "sglang" in cmdline:
                candidates.append(psutil_mod.Process(proc.info["pid"]))
        except (psutil_mod.NoSuchProcess, psutil_mod.AccessDenied):
            pass

    if not candidates:
        raise RuntimeError(
            "[sampler] FATAL: No SGLang server processes found. "
            "The server must be running before trace_run.py starts.\n"
            "Verify with:  ps aux | grep sglang\n"
            "If processes appear there but discovery still fails, update "
            "the match string in _discover_server_procs() to match the "
            "actual command-line tokens printed by that command."
        )

    pids = [p.pid for p in candidates]
    print(f"[sampler] found {len(candidates)} SGLang server processes: PIDs {pids}")
    for p in candidates:   # prime — first cpu_percent() always returns 0.0
        try:
            p.cpu_percent()
        except (psutil_mod.NoSuchProcess, psutil_mod.AccessDenied):
            pass
    return candidates


class HardwareSampler(threading.Thread):

    def __init__(self, interval: float):
        super().__init__(daemon=True, name="HardwareSampler")
        self.interval  = interval
        self.rows      = []
        self._stop_evt = threading.Event()
        self.hw_source = "none"

        # ── Try DCGM first ────────────────────────────────────────────────
        self._dcgm = None
        self.n_gpu = 0
        try:
            self._dcgm = DcgmCollector(update_freq_us=max(20_000, int(interval * 1_000_000 // 2)))
            self.n_gpu = self._dcgm.n_gpu
            self.hw_source = "dcgm"
            print(f"[sampler] using DCGM (SM-Act / SM-Occ / DRAM-Act)")
        except Exception as e:
            print(f"[sampler] DCGM unavailable ({e!r}); trying NVML fallback.")

        # ── NVML fallback ─────────────────────────────────────────────────
        self._nvml    = None
        self._handles = []
        if self._dcgm is None:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import pynvml
                pynvml.nvmlInit()
                self._nvml    = pynvml
                n = pynvml.nvmlDeviceGetCount()
                self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
                self.n_gpu    = n
                self.hw_source = "nvml"
                print(f"[sampler] using NVML fallback ({n} GPUs)")
            except Exception as e:
                print(f"[sampler] NVML also unavailable ({e}); GPU columns will be blank.")

        # ── CPU (psutil) ──────────────────────────────────────────────────
        import os, psutil
        self._psutil = psutil
        self.n_cpu   = psutil.cpu_count(logical=True)
        psutil.cpu_percent(percpu=True)   # prime whole-machine first call

        # ── Per-process CPU split (server group vs agent) ─────────────────
        self._agent_proc    = psutil.Process(os.getpid())
        self._agent_proc.cpu_percent()          # prime
        self._server_procs  = _discover_server_procs(psutil, os.getpid())
        self.server_pids    = [p.pid for p in self._server_procs]
        self.n_server_procs = len(self._server_procs)

        # ── disk I/O + host RAM (six-axis model: axis-3 disk, memory-capacity) ──
        # disk_io_counters() is cumulative -> we delta it vs the previous sample for a MB/s rate.
        # NOTE: system-wide reads count only ACTUAL disk hits, so a warm page-cache re-read of a
        # big file shows ~0 here (that page-cache effect is exactly what we want to see).
        try:
            self._prev_disk = psutil.disk_io_counters()
        except Exception:
            self._prev_disk = None
        self._prev_disk_t = now()

    def _read_gpu_dcgm(self) -> dict:
        """Returns {gpu_idx: {suffix: value}} via DCGM."""
        return self._dcgm.read()

    def _read_gpu_nvml(self) -> dict:
        """Returns {gpu_idx: {suffix: value}} via NVML, using canonical column names."""
        result = {}
        for i, h in enumerate(self._handles):
            row = {}
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(h)
                mem  = self._nvml.nvmlDeviceGetMemoryInfo(h)
                row["sm_act"]      = util.gpu          # NVML: % kernel active (≈ SM activity)
                row["sm_occ"]      = None              # not available from NVML
                row["dram_act"]    = util.memory       # NVML: % BW controller active (≈ DRAM activity)
                row["mem_used_mb"] = mem.used  // (1024 * 1024)
                row["mem_free_mb"] = mem.free  // (1024 * 1024)
            except Exception:
                for s in ("sm_act", "sm_occ", "dram_act", "mem_used_mb", "mem_free_mb"):
                    row[s] = None
            try:
                pw = self._nvml.nvmlDeviceGetPowerUsage(h)
                row["power_w"] = round(pw / 1000.0, 1)  # mW → W
            except Exception:
                row["power_w"] = None
            try:
                row["sm_clock_mhz"] = self._nvml.nvmlDeviceGetClockInfo(
                    h, self._nvml.NVML_CLOCK_SM)
            except Exception:
                row["sm_clock_mhz"] = None
            result[i] = row
        return result

    def run(self):
        while not self._stop_evt.is_set():
            t = now()
            row = {"t": t, "hw_source": self.hw_source}

            if self._dcgm is not None:
                gpu_data = self._read_gpu_dcgm()
            elif self._nvml is not None:
                gpu_data = self._read_gpu_nvml()
            else:
                gpu_data = {}

            for i in range(self.n_gpu):
                gd = gpu_data.get(i, {})
                for s in _GPU_SUFFIXES:
                    row[f"gpu{i}_{s}"] = gd.get(s)

            per = self._psutil.cpu_percent(percpu=True)
            for i, v in enumerate(per):
                row[f"cpu{i}"] = v
            row["cpu_mean"]      = round(sum(per) / len(per), 3) if per else None
            row["cpu_max"]       = round(max(per), 1)            if per else None
            row["cpu_hot_cores"] = sum(1 for v in per if v > 50.0)

            # per-process split: server group and agent process
            srv = 0.0
            for p in self._server_procs:
                try:
                    srv += p.cpu_percent()
                except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                    pass
            try:
                agt = self._agent_proc.cpu_percent()
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                agt = None
            row["proc_server_cpu"] = round(srv, 2)
            row["proc_agent_cpu"]  = round(agt, 2) if agt is not None else None

            # ── disk I/O throughput (system-wide, rate from cumulative counters) ──
            try:
                d  = self._psutil.disk_io_counters()
                dt = t - self._prev_disk_t
                if d is not None and self._prev_disk is not None and dt > 0:
                    row["disk_read_mbps"]  = round((d.read_bytes  - self._prev_disk.read_bytes)  / dt / 1e6, 2)
                    row["disk_write_mbps"] = round((d.write_bytes - self._prev_disk.write_bytes) / dt / 1e6, 2)
                    busy_ms = (d.read_time - self._prev_disk.read_time) + (d.write_time - self._prev_disk.write_time)
                    row["disk_busy_pct"]   = round(min(100.0, max(0.0, busy_ms) / (dt * 1000.0) * 100.0), 1)
                else:
                    row["disk_read_mbps"] = row["disk_write_mbps"] = row["disk_busy_pct"] = None
                if d is not None:
                    self._prev_disk = d; self._prev_disk_t = t
            except Exception:
                row["disk_read_mbps"] = row["disk_write_mbps"] = row["disk_busy_pct"] = None

            # ── host RAM (OOM cliff + big-data load footprint) ──
            try:
                vm = self._psutil.virtual_memory()
                row["ram_used_gb"]  = round(vm.used / 1e9, 2)
                row["ram_avail_gb"] = round(vm.available / 1e9, 2)
                row["ram_pct"]      = vm.percent
            except Exception:
                row["ram_used_gb"] = row["ram_avail_gb"] = row["ram_pct"] = None

            self.rows.append(row)
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        if self._dcgm is not None:
            self._dcgm.shutdown()

    def measure_server_idle_baseline(self, duration_s: float = 3.0) -> float | None:
        """
        Measure the SGLang server's idle CPU floor over duration_s seconds.

        Creates separate psutil.Process handles (independent of the background
        thread's handles) so cpu_percent() intervals don't interfere. Returns
        the mean total server CPU % (sum across all server processes).

        Call after .start() but before agent.go().
        """
        import time
        # Separate Process objects — independent tracking from background thread
        baseline_procs = []
        for p in self._server_procs:
            try:
                bp = self._psutil.Process(p.pid)
                bp.cpu_percent()   # prime
                baseline_procs.append(bp)
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                pass
        if not baseline_procs:
            return None

        interval = 0.5
        n = max(2, int(duration_s / interval))
        readings = []
        for _ in range(n):
            time.sleep(interval)
            total = 0.0
            for bp in baseline_procs:
                try:
                    total += bp.cpu_percent()
                except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                    pass
            readings.append(total)

        baseline = round(sum(readings) / len(readings), 2) if readings else None
        print(f"[sampler] server idle CPU baseline: {baseline}%  "
              f"({n} samples × {interval}s, {len(self._server_procs)} processes)")
        return baseline

    def dump(self, path: Path):
        if not self.rows:
            return
        keys = ["t", "hw_source"]
        keys += [f"gpu{i}_{s}" for i in range(self.n_gpu) for s in _GPU_SUFFIXES]
        keys += [f"cpu{i}" for i in range(self.n_cpu)]
        keys += ["cpu_mean", "cpu_max", "cpu_hot_cores"]
        keys += ["proc_server_cpu", "proc_agent_cpu"]
        keys += ["disk_read_mbps", "disk_write_mbps", "disk_busy_pct", "ram_used_gb", "ram_avail_gb", "ram_pct"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in self.rows:
                w.writerow({k: ("" if r.get(k) is None else r[k]) for k in keys})


# ─────────────────────────────────────────────────────────────────────────────
#  Summary: derive structure from the event timeline (Layer-1 only).
# ─────────────────────────────────────────────────────────────────────────────
def summarize(events, agent_go_wall_s):
    gen_spans, exe_spans = [], []
    open_gen = open_exe = None
    for e in events:
        if e["phase"] == "generate":
            if e["edge"] == "start":
                open_gen = e["t"]
            elif open_gen is not None:
                gen_spans.append((open_gen, e["t"]))
                open_gen = None
        elif e["phase"] == "execute":
            if e["edge"] == "start":
                open_exe = e["t"]
            elif open_exe is not None:
                exe_spans.append((open_exe, e["t"]))
                open_exe = None

    gen_t = sum(b - a for a, b in gen_spans)
    exe_t = sum(b - a for a, b in exe_spans)

    return {
        "n_generate_phases": len(gen_spans),
        "n_execute_phases":  len(exe_spans),
        "turns": min(len(gen_spans), len(exe_spans)) if exe_spans else len(gen_spans),
        "generate_time_s":                   round(gen_t, 3),
        "execute_time_s":                    round(exe_t, 3),
        "gpu_bubble_time_s_layer1_estimate": round(exe_t, 3),
        "wall_time_s":                       round(agent_go_wall_s, 3),
        "generate_spans": [(round(a, 3), round(b, 3)) for a, b in gen_spans],
        "execute_spans":  [(round(a, 3), round(b, 3)) for a, b in exe_spans],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Run one Biomni agent task and capture Layer-1..4 traces."
    )
    tg = ap.add_mutually_exclusive_group()
    tg.add_argument("--task-config", metavar="JSON",
                    help="Path to a task config file in tasks/. Preferred.")
    tg.add_argument("--task", metavar="PROMPT",
                    help="Task prompt string directly (no config file).")
    ap.add_argument("--base-url",    default="http://localhost:30000/v1")
    ap.add_argument("--model",       default="RyanLi0802/Biomni-R0-Preview")
    ap.add_argument("--data-path",   default="./data")
    ap.add_argument("--hw-interval", type=float, default=0.05,
                    help="Hardware + SGLang sampling interval in seconds (default 0.05 = 50 ms).")
    ap.add_argument("--output-dir",       default="results", dest="outdir",
                    help="Root output directory (default: results/). Override to "
                         "results_multi/session_X/agentY/ for concurrent runs.")
    ap.add_argument("--concurrent-session", default=None, dest="concurrent_session",
                    help="Session ID written into meta.json to link agents in the "
                         "same concurrent run (e.g. 'session_2026-07-10_14-23-45'). "
                         "Omit for normal single-agent runs.")
    args = ap.parse_args()

    # ── Load task config ──────────────────────────────────────────────────
    task_cfg = {}
    if args.task_config:
        with open(args.task_config) as f:
            task_cfg = json.load(f)
        print(f"[trace] task config: {args.task_config}")
    elif not args.task:
        ap.error("Provide either --task-config tasks/xxx.json or --task 'prompt text'.")

    task_prompt               = args.task or task_cfg["prompt"]
    task_id                   = task_cfg.get("task_id")
    task_family               = task_cfg.get("task_family")
    resource_profile_expected = task_cfg.get("resource_profile_expected")
    expected_data_lake_files  = task_cfg.get("expected_data_lake_files", [])
    use_tool_retriever        = task_cfg.get("use_tool_retriever", True)
    timeout_seconds           = task_cfg.get("timeout_seconds", 600)

    # ── Output directory ──────────────────────────────────────────────────
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_dir = Path(args.outdir) / (task_id or "") / stamp if task_id else Path(args.outdir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "trace.log"
    tee_out, tee_err = _start_logging(log_path)

    print(f"[trace] output    → {run_dir}")
    print(f"[trace] log       → {log_path}")
    print(f"[trace] hw-interval = {args.hw_interval}s")
    print(f"[trace] task:  {task_prompt[:80]}{'...' if len(task_prompt) > 80 else ''}")

    # ── Point Biomni at local SGLang server ───────────────────────────────
    from biomni.config import default_config
    default_config.source   = "Custom"
    default_config.base_url = args.base_url
    default_config.api_key  = "EMPTY"
    default_config.llm      = args.model

    # ── Install execute-phase hooks BEFORE the agent is built ─────────────
    _wrap_execute()

    # ── Build the agent ───────────────────────────────────────────────────
    from biomni.agent import A1
    agent = A1(
        path=args.data_path,
        llm=args.model,
        source="Custom",
        base_url=args.base_url,
        api_key="EMPTY",
        expected_data_lake_files=expected_data_lake_files,
        use_tool_retriever=use_tool_retriever,
        timeout_seconds=timeout_seconds,
    )

    # ── Hook the generate phase (Layer-1 + Layer-2) ───────────────────────
    hooked = False
    for attr in ("llm", "model", "_llm", "chat_model"):
        obj = getattr(agent, attr, None)
        if obj is not None and hasattr(obj, "invoke"):
            _wrap_generate(obj)
            hooked = True
            break
    if not hooked:
        print("[wrap] could not locate agent LLM object; generate phase not hooked.")

    # ── Start samplers ────────────────────────────────────────────────────
    # In concurrent mode the LAUNCHER runs ONE shared L3+L4 sampler for the whole
    # session; per-agent samplers are skipped so N agents don't each hammer /metrics
    # (that self-inflicted load added ~34s to every call — see the 2026-08-07 diagnosis).
    concurrent = bool(args.concurrent_session)
    if concurrent:
        hw_sampler = None
        sglang_scraper = None
        server_idle_baseline = None
        print("[trace] concurrent mode: L3/L4 sampling + analysis delegated to the shared session sampler.")
    else:
        hw_sampler    = HardwareSampler(interval=args.hw_interval)
        sglang_scraper = SGLangScraper(base_url=args.base_url, interval=args.hw_interval)
        hw_sampler.start()
        sglang_scraper.start()
        # Measure server idle CPU before the agent starts (baseline for attribution)
        server_idle_baseline = hw_sampler.measure_server_idle_baseline(duration_s=3.0)

    # ── Run one agent program ─────────────────────────────────────────────
    print(f"\n[trace] starting agent.go() ...\n")
    err = None
    t_start = now()
    try:
        result = agent.go(task_prompt)
    except Exception as e:
        err = repr(e)
        result = None
        print(f"[trace] agent.go raised: {err}")
    t_end = now()
    agent_go_wall_s = t_end - t_start

    # ── Stop samplers ─────────────────────────────────────────────────────
    if not concurrent:
        hw_sampler.stop()
        sglang_scraper.stop()
        hw_sampler.join(timeout=2 * args.hw_interval + 1)
        sglang_scraper.join(timeout=2 * args.hw_interval + 1)

    # ── Write all outputs ─────────────────────────────────────────────────
    # hardware.csv / sglang_metrics.csv are written per-agent only in single mode; in
    # concurrent mode the launcher reprojects the shared session CSVs into this folder.
    outputs = [("events.jsonl", lambda: EVENTS.dump(run_dir / "events.jsonl"))]
    if not concurrent:
        outputs += [
            ("hardware.csv",       lambda: hw_sampler.dump(run_dir / "hardware.csv")),
            ("sglang_metrics.csv", lambda: sglang_scraper.dump(run_dir / "sglang_metrics.csv")),
        ]
    for label, fn in outputs:
        try:
            fn()
        except Exception as e:
            print(f"[trace] WARNING: failed to write {label}: {e!r}")

    summary = summarize(EVENTS.events, agent_go_wall_s)
    summary["agent_go_wall_s"] = round(agent_go_wall_s, 3)
    summary["error"]           = err
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    meta = {
        "task":                      task_prompt,
        "task_id":                   task_id,
        "task_family":               task_family,
        "resource_profile_expected": resource_profile_expected,
        "task_config_file":          args.task_config,
        "model":                     args.model,
        "base_url":                  args.base_url,
        "hw_interval_s":             args.hw_interval,
        "hw_source":                 ("shared_session_sampler" if concurrent else hw_sampler.hw_source),
        "n_gpu_sampled":             (None if concurrent else hw_sampler.n_gpu),
        "n_cpu_sampled":             (None if concurrent else hw_sampler.n_cpu),
        "t0_wall_unix":              T0_WALL,
        "t0_wall_iso":               dt.datetime.fromtimestamp(
                                         T0_WALL, dt.timezone.utc).isoformat(),
        "concurrent_session":        args.concurrent_session,
        "server_n_procs":            (None if concurrent else hw_sampler.n_server_procs),
        "server_pids":               (None if concurrent else hw_sampler.server_pids),
        "server_idle_cpu_baseline":  server_idle_baseline,
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if result is not None:
        try:
            (run_dir / "agent_output.txt").write_text(str(result))
        except Exception:
            pass

    print("\n[trace] done.")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("generate_spans", "execute_spans")}, indent=2))

    # Auto-analyze: generate analysis.json + all figures without a separate manual step.
    # Skipped in concurrent mode — the launcher reprojects the shared session CSVs into
    # this folder and then runs analyze_trace.py itself (the CSVs don't exist yet here).
    if concurrent:
        print("[trace] concurrent mode: hardware.csv/sglang_metrics.csv + analysis are produced by the launcher.")
    else:
        print(f"\n[trace] auto-running analyze_trace.py on {run_dir} ...")
        try:
            import subprocess
            subprocess.run(
                [sys.executable,
                 str(Path(__file__).parent / "analyze_trace.py"),
                 str(run_dir)],
                check=True,
            )
        except Exception as e:
            print(f"[trace] WARNING: analyze_trace.py failed: {e!r}")
            print(f"        Re-run manually: python profiling/analyze_trace.py {run_dir}")

    _stop_logging(tee_out, tee_err)


if __name__ == "__main__":
    main()
