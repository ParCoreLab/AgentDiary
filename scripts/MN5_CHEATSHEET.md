# MN5 cheat sheet — run traces on MareNostrum 5 without help

## Mental model (read once)
- **Login node** = where you type. `alogin2.bsc.es` (GPU/ACC side) for running jobs;
  `glogin1.bsc.es` (GPP side) only for the internet tunnel / pip installs. **Login nodes have NO GPU.**
- **Compute node** = where your job actually runs (4× H100). You NEVER ssh here — SLURM picks one for you.
- **`sbatch`** = drop a job in the queue. SLURM starts it when a GPU node frees up.
- **/gpfs is shared** across all MN5 nodes: edit a file once, every node sees it.
  **du04 is a SEPARATE machine** — you must `rsync` files between du04 and MN5 by hand.
- Project root on MN5: `/gpfs/projects/etur02/koc858886/biomni`  (call it `$B`)
  - repo:  `$B/biomni-profiling`   scripts: `$B/biomni-profiling/scripts`   results: `$B/biomni-profiling/results`
  - logs:  `$B/sweep_<JOBID>.log`

## The 4 commands you use every time
```bash
# 1. get on the GPU login node (from du04 or your laptop)
ssh alogin2.bsc.es                       # (alogin1 if alogin2 is down — they're twins)

# 2. submit a sweep: load server ONCE, run each task N_REPS times
cd /gpfs/projects/etur02/koc858886/biomni/biomni-profiling/scripts
sbatch --qos=acc_debug --time=02:00:00 mn5_sweep.sh 1 <task_id> [task_id2 ...]   # quick 2h test
sbatch mn5_sweep.sh 20 <task_id>                                                # real run (default 24h queue)

# 3. watch the queue                     PD = waiting for a node, R = running, CG = finishing
squeue --me

# 4. watch the run live (JOBID from squeue)
tail -f /gpfs/projects/etur02/koc858886/biomni/sweep_<JOBID>.log
```

## Anatomy of the sweep command
`sbatch [slurm-opts] mn5_sweep.sh  N_REPS  TASK1 TASK2 ...`
- **N_REPS** — run EACH task this many times. `1` = just test it works; `20` = the real distribution.
- **TASK\*** — task_id = the filename in `tasks/` without `.json`.
- Default queue is `acc_ehpc` (24 h, big sweeps). For a quick test override with `--qos=acc_debug --time=02:00:00`.
- The ~7-min model load happens ONCE per job; every rep/task runs against the same server.

## After I (Claude) edit tasks or code on du04 → SYNC to MN5 first
Otherwise MN5 runs the OLD files. Run these FROM du04:
```bash
# tasks (the usual one). --delete makes MN5 mirror du04 exactly (moves archived files correctly)
rsync -av --delete /home/mansari26/biomni-profiling/tasks/ \
  koc858886@glogin1.bsc.es:/gpfs/projects/etur02/koc858886/biomni/biomni-profiling/tasks/

# if profiling code changed, swap tasks/ -> profiling/  (do NOT use --delete on profiling/ — no)
rsync -av /home/mansari26/biomni-profiling/profiling/ \
  koc858886@glogin1.bsc.es:/gpfs/projects/etur02/koc858886/biomni/biomni-profiling/profiling/
```

## Get results back to du04 (to view figures/CSV in VSCode). Run FROM du04:
```bash
rsync -av koc858886@glogin1.bsc.es:/gpfs/projects/etur02/koc858886/biomni/biomni-profiling/results/ \
  ~/biomni-profiling/results_mn5/
# then open results_mn5/<task_id>/<timestamp>/fig_overview.png  and  results_mn5/aggregate.csv
```

## Handy extras
```bash
scancel <JOBID>                          # kill a job
squeue --me -o "%.18i %.9P %.8j %.2t %R" # %R = why pending (Priority/Resources = just waiting for a node)
bsc_quota                                # storage left
# submit from du04 without an interactive login:
ssh alogin2.bsc.es "cd /gpfs/projects/etur02/koc858886/biomni/biomni-profiling/scripts && \
  sbatch --qos=acc_debug --time=02:00:00 mn5_sweep.sh 1 nmf_tumor_subtypes"
```

## Reading the result
- `results/<task_id>/<ts>/analysis.json` → per-run metrics (bubble, execute_total_s, ...).
- `results/aggregate.csv` → one row per run. **`agent_completed` column**: `True` = real run
  (use its bubble); `False` = the model failed/ran away (counts toward the failure rate, ignore its bubble).
- A task is "good" if it completes with `execute_total_s` well above 5 s.
