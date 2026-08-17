# mn5_jtail.sh — tail a SLURM job's log without knowing its filename prefix.
# Source this once (add to ~/.bashrc on MN5):   source /gpfs/projects/etur02/koc858886/biomni/biomni-profiling/scripts/mn5_jtail.sh
#
#   jtail            tail -f your NEWEST job's log (running or just-finished)
#   jtail <JOBID>    tail -f that specific job's log
#   jlog  [JOBID]    just print the exact log path (e.g. to grep it)
#
# It asks SLURM directly (scontrol StdOut=), so the sweep_/concurrent_/bench_ prefix
# never matters. Falls back to globbing *_<JOBID>.log for older finished jobs.

_biomni_joblog() {
  local jid="${1:-$(squeue --me -h -o %i 2>/dev/null | sort -rn | head -1)}"
  [ -z "$jid" ] && { echo "jtail: no running job — pass a JOBID" >&2; return 1; }
  local f
  f=$(scontrol show job "$jid" 2>/dev/null | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -1)
  [ -z "$f" ] && f=$(ls -t /gpfs/projects/etur02/koc858886/biomni/*_"$jid".log 2>/dev/null | head -1)
  [ -z "$f" ] && { echo "jtail: no log found for job $jid" >&2; return 1; }
  printf '%s\n' "$f"
}
jlog()  { _biomni_joblog "$@"; }
jtail() { local f; f=$(_biomni_joblog "$@") || return 1; echo "==> $f  (Ctrl-C to stop)"; tail -f "$f"; }
