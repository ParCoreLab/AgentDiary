#!/usr/bin/env python3
"""
Tidy multi-agent session dirs: move every per-agent `agent_*` folder into an `agents/` subdir, so the
session-level deliverables (session_hardware.csv, session_sglang_metrics.csv, session_summary.json,
session_comparison.txt, fig_concurrent_*.png) sit ALONE at the top of the session for easy access.

Idempotent + safe: only moves agent_* that are at the session top level; skips already-organized sessions;
never overwrites an existing agents/agent_* (warns instead).

  python scripts/organize_sessions.py                 # default roots: results_mn5_multi/ + results_multi/
  python scripts/organize_sessions.py <root_or_session> ...
"""
import sys, glob, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = [REPO / "results_mn5_multi", REPO / "results_multi"]


def is_session(d: Path):
    return d.is_dir() and (
        (d / "session_summary.json").exists() or (d / "session_comparison.txt").exists()
        or (d / "session_config.json").exists() or any(d.glob("agent_*"))
    )


def session_dirs(root: Path):
    root = Path(root)
    if is_session(root):
        return [root]
    # sessions live at <root>/session_* and <root>/<sub>/session_* (e.g. throttle_120c/session_*)
    found = set()
    for pat in ("session_*", "*/session_*"):
        for p in glob.glob(str(root / pat)):
            if is_session(Path(p)):
                found.add(Path(p))
    return sorted(found)


def organize(session: Path):
    top_agents = [d for d in session.glob("agent_*") if d.is_dir()]
    if not top_agents:
        return 0
    dest = session / "agents"
    dest.mkdir(exist_ok=True)
    moved = 0
    for ad in top_agents:
        target = dest / ad.name
        if target.exists():
            print(f"    !! skip {ad.name}: already exists in agents/"); continue
        shutil.move(str(ad), str(target)); moved += 1
    return moved


def main():
    roots = [Path(a) for a in sys.argv[1:]] or DEFAULT_ROOTS
    sessions = []
    for r in roots:
        if r.exists():
            sessions += session_dirs(r)
    if not sessions:
        print("no session dirs found under", [str(r) for r in roots]); return
    total = 0
    for s in sorted(set(sessions)):
        n = organize(s)
        if n:
            total += n
            print(f"  {s.relative_to(REPO) if str(s).startswith(str(REPO)) else s}: moved {n} agent folders -> agents/")
    print(f"done — organized {total} agent folders across {len(set(sessions))} sessions "
          f"(top level now holds only session_*.{{csv,json,txt}} + figures + agents/)")


if __name__ == "__main__":
    main()
