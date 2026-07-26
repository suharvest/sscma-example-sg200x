#!/bin/sh
# Sample RSS / FD / thread counts for the two long-lived processes. Copy to the
# device and run there; RECAMERA_PW must be set for the sudo needed to list
# another user's file descriptors.
S() { echo "$RECAMERA_PW" | sudo -S "$@" 2>/dev/null; }
SV=$(ps -o pid,args 2>/dev/null | grep "[s]upervisor -g" | awk '{print $1}' | head -1)
FA=$(ps -o pid,args 2>/dev/null | grep -E "[f]ace-analysis|[y]olo-detector|[q]rcode-reader" | awk '{print $1}' | head -1)
echo "T=$(date '+%H:%M:%S') mem_avail=$(free -m | awk '/^Mem/{print $7}')MB"
for pair in "$SV supervisor" "$FA app"; do
  set -- $pair; pid=$1; name=$2
  [ -n "$pid" ] && [ -d /proc/$pid ] || { echo "  $name: not running"; continue; }
  rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status)
  thr=$(awk '/Threads/{print $2}' /proc/$pid/status)
  fds=$(S ls /proc/$pid/fd | wc -l)
  printf "  %-14s pid=%-6s RSS=%-8s FD=%-5s threads=%s\n" "$name" "$pid" "${rss}KB" "$fds" "$thr"
done
