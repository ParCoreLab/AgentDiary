#!/bin/bash
# Copy Biomni-R0 (~123 GB) from du04 -> MN5 /gpfs, via BSC's transfer node.
# RESUMABLE: if interrupted, just run this again — rsync continues where it left off.
# Watch progress with:  tail -f ~/model_copy.log
set -o pipefail

SRC="/DATA/mansari26/huggingface_cache/hub/models--RyanLi0802--Biomni-R0-Preview"
DEST_HOST="koc858886@transfer1.bsc.es"
DEST_DIR="/gpfs/projects/etur02/koc858886/biomni/hf_cache/hub"

echo "=================================================================="
echo "=== model copy started: $(date) ==="
echo "source: $SRC"
echo "dest:   $DEST_HOST:$DEST_DIR"
echo "=================================================================="

# 1. make sure the destination folder exists on MN5
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$DEST_HOST" "mkdir -p '$DEST_DIR'" \
  || { echo "ERROR: cannot reach transfer node $DEST_HOST"; exit 1; }

# 2. transfer, with auto-retry. Each attempt RESUMES via --partial --append-verify.
#    SSH keepalive (ServerAliveInterval) reduces mid-transfer drops on the slow link.
#    No -z: model weights don't compress. Retries up to 30x, resuming each time.
RC=1
for attempt in $(seq 1 30); do
  echo "--- rsync attempt $attempt at $(date) ---"
  rsync -a --partial --append-verify --info=progress2 --human-readable \
    -e "ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o BatchMode=yes" \
    "$SRC" "$DEST_HOST:$DEST_DIR/"
  RC=$?
  if [ "$RC" -eq 0 ]; then echo "rsync completed on attempt $attempt"; break; fi
  echo "attempt $attempt ended rc=$RC (likely a dropped connection) — resuming in 15s..."
  sleep 15
done

echo ""
echo "=== rsync finished rc=$RC at $(date) ==="
if [ "$RC" -eq 0 ]; then
  echo "=== size now on MN5 (should be ~123G) ==="
  ssh -o BatchMode=yes "$DEST_HOST" "du -sh '$DEST_DIR/models--RyanLi0802--Biomni-R0-Preview'"
fi
echo "=== DONE rc=$RC ==="
