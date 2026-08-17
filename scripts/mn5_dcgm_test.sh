#!/bin/bash
# Confirm the profiling fields populate via DcgmReader with trace_run.py's exact params.
DCGM=/gpfs/projects/etur02/koc858886/biomni/dcgm
export LD_LIBRARY_PATH="$DCGM/usr/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$DCGM/usr/bin:$PATH"
export DCGM_BINDINGS_PATH="$DCGM/usr/share/datacenter-gpu-manager-4/bindings/python3"
HELOG=/gpfs/projects/etur02/koc858886/biomni/he_run.log
module load anaconda/2024.02
source /gpfs/projects/etur02/koc858886/biomni/venv/bin/activate

echo "node: $(hostname)"
nv-hostengine -n > "$HELOG" 2>&1 &
HE_PID=$!
sleep 7
kill -0 "$HE_PID" 2>/dev/null && echo "hostengine ALIVE" || { echo DIED; tail -8 "$HELOG"; }

python3 -c "
import sys, os, time
sys.path.insert(0, os.environ['DCGM_BINDINGS_PATH'])
from DcgmReader import DcgmReader
r = DcgmReader(hostname='localhost', fieldIds=[1002,1003,1005,155,252],
               updateFrequency=50000, maxKeepAge=300.0,
               fieldGroupName='biomni_test_pg', ignoreBlank=True)
r.Init()
for i in range(6):
    time.sleep(0.5)
    v = r.GetLatestGpuValuesAsFieldIdDict()
    g = sorted(v.keys())[0]
    print(f'  read {i}: SMact={v[g].get(1002)} SMocc={v[g].get(1003)} DRAM={v[g].get(1005)} power={v[g].get(155)}')
r.Shutdown()
" 2>&1 | tail -8
kill "$HE_PID" 2>/dev/null
echo "=== done ==="
