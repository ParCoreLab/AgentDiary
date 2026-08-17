#!/bin/bash
# Durable reverse-SOCKS tunnel: du04 -> MareNostrum 5 (login node glogin1).
# glogin1 allows reverse tunnels (the ACC login alogin1 blocks them); the /gpfs
# disk is shared, so software downloaded via glogin1 is visible to the GPU nodes.
# On MN5 glogin1, point tools at the proxy with:  socks5h://localhost:18080
#
# Usage:  bash mn5_tunnel.sh {start|status|stop}
PORT=18080
MN5="koc858886@glogin1.bsc.es"
# Host-agnostic kill/status pattern so 'start' also clears an old tunnel to any MN5 login node.
PAT="R ${PORT} koc858886@"

case "$1" in
  start)
    pkill -f "$PAT" 2>/dev/null
    nohup autossh -M 0 -N -R ${PORT} ${MN5} \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new \
      > ~/mn5_tunnel.log 2>&1 &
    disown
    sleep 4
    bash "$0" status
    ;;
  status)
    if pgrep -af "$PAT" | grep -q autossh; then
      echo "TUNNEL RUNNING:"
      pgrep -af autossh | grep "$PORT"
    else
      echo "TUNNEL NOT RUNNING"
      echo "--- last lines of ~/mn5_tunnel.log ---"
      tail -5 ~/mn5_tunnel.log 2>/dev/null
    fi
    ;;
  stop)
    pkill -f "$PAT" && echo "tunnel stopped" || echo "tunnel was not running"
    ;;
  *)
    echo "usage: bash $0 {start|status|stop}"
    ;;
esac
