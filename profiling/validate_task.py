#!/usr/bin/env python3
"""
Task-trust checks, so a task is VERIFIED (not assumed) before and after a sweep.
Two failure modes this catches, both bit us on BindingDB:
  1. a dependency missing on the TARGET machine (rdkit absent on MN5) -> the agent flails/skips it.
  2. a "hollow" completion: error=None but the intended computation never happened (no fingerprints,
     garbage R2). error=None is NOT proof the science ran.

A task JSON may carry two optional fields:
  "requires":         ["rdkit", "sklearn", ...]                  # python modules the task needs
  "success_markers":  ["computed fingerprints for", "R2"]        # regexes that MUST appear in a VALID run's trace.log

Usage:
  # BEFORE sweeping, on the machine that will run it (checks imports actually work there):
  python profiling/validate_task.py --deps tasks/bindingdb_egfr_qsar.json [more.json ...]

  # AFTER runs, filter valid vs hollow (task_id is taken from the results/<task_id>/... path):
  python profiling/validate_task.py --run results/bindingdb_egfr_qsar/*/
"""
import sys, os, json, glob, re, importlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def check_deps(task_files):
    allok = True
    for tf in task_files:
        t = json.load(open(tf))
        req = t.get("requires", [])
        print(f"[deps] {t['task_id']}: requires {req or '(none listed)'}")
        for mod in req:
            try:
                importlib.import_module(mod)
                print(f"    ok       {mod}")
            except Exception as e:
                print(f"    MISSING  {mod}  ({type(e).__name__})")
                allok = False
    print("[deps] ALL PRESENT" if allok else "[deps] *** MISSING DEPS — do NOT sweep on this machine ***")
    return allok


def _markers_for(task_id, field="success_markers"):
    tf = REPO / "tasks" / f"{task_id}.json"
    if not tf.exists():
        for alt in glob.glob(str(REPO / "tasks" / "**" / f"{task_id}.json"), recursive=True):
            tf = Path(alt); break
    try:
        return json.load(open(tf)).get(field, [])
    except Exception:
        return []


def check_run(run_dirs):
    valid = hollow = 0
    for d in run_dirs:
        d = Path(d)
        if not (d / "trace.log").exists():
            continue
        # task_id = the results/<task_id>/<ts>/ parent
        task_id = d.parent.name
        markers = _markers_for(task_id)
        reqs = _markers_for(task_id, "requires")
        log = (d / "trace.log").read_text(errors="ignore")
        out = (d / "agent_output.txt").read_text(errors="ignore") if (d / "agent_output.txt").exists() else ""
        # Strip the agent's CODE (<execute> blocks) so markers match OUTPUT/observations, not the code:
        # a run can print "...computed fingerprints for..." in code yet error (e.g. rdkit missing) before it runs.
        blob = re.sub(r"<execute>.*?</execute>", "", log, flags=re.S) + out
        missing = [m for m in markers if not re.search(m, blob)]
        err = None
        try:
            err = json.loads((d / "summary.json").read_text()).get("error")
        except Exception:
            pass
        dep_fail = [m for m in reqs if re.search(r"No module named ['\"]?" + re.escape(m), log)]
        if err is not None and str(err) not in ("", "None"):
            verdict = f"FAILED (error: {str(err)[:34]})"
        elif dep_fail:
            verdict = f"HOLLOW (required dep missing at runtime: {dep_fail})"; hollow += 1
        elif not markers:
            verdict = "NO-MARKERS (add success_markers)"
        elif missing:
            verdict = f"HOLLOW (missing {missing})"; hollow += 1
        else:
            verdict = "VALID"; valid += 1
        print(f"  {d.parent.name}/{d.name}: error={err!s:6.6s}  {verdict}")
    print(f"\n[run] VALID {valid} | HOLLOW {hollow}  (VALID = error is None AND every success_marker present)")
    return hollow == 0


def main():
    a = sys.argv[1:]
    if a and a[0] == "--deps":
        sys.exit(0 if check_deps(a[1:]) else 1)
    if a and a[0] == "--run":
        sys.exit(0 if check_run(a[1:]) else 1)
    print(__doc__)


if __name__ == "__main__":
    main()
