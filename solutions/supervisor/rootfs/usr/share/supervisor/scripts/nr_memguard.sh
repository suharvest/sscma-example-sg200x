#!/bin/sh
#
# nr_memguard.sh — RSS watchdog for the Node-RED / sscma-node stack (#19,
# Method A). This device has 256MB RAM (~181MB usable) + 256MB swap and NO
# cgroups (CONFIG_CGROUPS unset), so a runaway Node-RED flow can thrash RAM and
# swap and starve the maintenance channel. The old `ulimit -v` cap was the
# wrong tool: it bounds VSZ (virtual address space), and V8/node reserve far
# more VSZ than they ever resident-use, so a cap tight enough to matter for
# physical RAM stopped node-red from starting. This watchdog instead samples
# real RESIDENT private memory (/proc/<pid>/RssAnon; see the metric note below)
# and kills the node stack when its total exceeds the threshold.
#
# --- Coordination with the serviced watchdog (IMPORTANT) --------------------
# In nodered mode the supervisor's in-process `serviced` watchdog monitors
# node-red and RESTARTS it if it dies. So when nr_memguard kills an over-budget
# node stack, serviced brings up a FRESH, low-RSS node-red. Steady state:
# node-red sits comfortably under the threshold and the watchdog never fires.
# Pathological state ("a flow leaks without bound"): node-red膨胀 -> killed ->
# restarted -> 膨胀 -> killed ... i.e. it FLAPS. That flap is deliberate and
# acceptable: every restart releases the leaked memory, so the device NEVER
# slides into the swap-thrash death spiral that the old ulimit-wedged build
# fell into. Flapping node-red beats a bricked device; the maintenance/HTTP
# channel (choom -900/-800 on supervisor/sshd) always survives.
#
# --- Metric: RssAnon, not VmRSS (measured on real hardware, 2026-07) --------
# We sample RssAnon (private anonymous pages = the JS heap / native Buffer
# growth a leaking flow actually produces), NOT VmRSS. VmRSS also counts
# shared-library and file-backed (mmap'd .js/.node) pages, which are reclaimable
# and hugely inflate the number: a single HEALTHY node-red measures ~90MB VmRSS
# steady and SPIKES to ~172MB VmRSS during startup, while its private heap
# (RssAnon) is only ~56MB. A VmRSS cap of 150MB therefore killed a perfectly
# healthy node-red mid-startup (1880 never came up). RssAnon tracks real
# growth and leaves generous headroom under the same threshold. A 2-strike
# debounce (kill only after two consecutive over-threshold samples) further
# prevents a transient spike from tripping the guard.
#
# --- Lifecycle --------------------------------------------------------------
# Started ONLY in nodered mode: by main.sh setRunMode->nodered (after node-red
# is up + mode persisted) and by S93 at boot when the persisted mode is
# nodered. Stopped on every switch back to console (setRunMode->console,
# forceConsole) and at console boot, via nr_memguard_stop (pid file). As a
# second line of defense this loop ALSO self-exits if it observes the mode file
# flip away from "nodered", so a missed kill cannot leave a stale watchdog
# running in console mode. main.sh (re)launches it idempotently (kills the old
# instance before starting a new one).
#
# --- Threshold --------------------------------------------------------------
# NR_RSS_MAX_KB (KiB), default 150000 (~146MB). Override precedence:
#   1. exported NR_RSS_MAX_KB env var
#   2. /etc/recamera.conf/nr_rss_max_kb  (plain integer; CONFIG_DIR is a
#      directory on this BSP)
# A non-integer / empty value falls back to the default.

PIDFILE=/var/run/nr_memguard.pid
LOGFILE=/var/log/nr_memguard.log
MODE_FILE=/userdata/local/apps/mode
CONFIG_DIR=/etc/recamera.conf
CONFIG_FILE="$CONFIG_DIR/nr_rss_max_kb"
INTERVAL=5
DEFAULT_MAX_KB=150000

_log() {
    # Best-effort, size-bounded log (keep the last ~200 lines).
    echo "$(date '+%Y-%m-%d %H:%M:%S') nr_memguard[$$]: $*" >>"$LOGFILE" 2>/dev/null || true
    if [ -f "$LOGFILE" ]; then
        lines=$(wc -l <"$LOGFILE" 2>/dev/null || echo 0)
        [ "${lines:-0}" -gt 400 ] 2>/dev/null && {
            tail -n 200 "$LOGFILE" >"$LOGFILE.tmp" 2>/dev/null && mv -f "$LOGFILE.tmp" "$LOGFILE" 2>/dev/null || true
        }
    fi
}

_max_kb() {
    max=""
    if [ -n "$NR_RSS_MAX_KB" ]; then
        max="$NR_RSS_MAX_KB"
    elif [ -f "$CONFIG_FILE" ]; then
        max=$(head -n 1 "$CONFIG_FILE" 2>/dev/null | tr -dc '0-9')
    fi
    case "$max" in
    '' | *[!0-9]*) max="$DEFAULT_MAX_KB" ;;
    esac
    [ "$max" -gt 0 ] 2>/dev/null || max="$DEFAULT_MAX_KB"
    echo "$max"
}

# Space-separated pids of the whole node stack. pidof matches exact comm names,
# so the sets are disjoint and need no dedup.
_node_pids() {
    pidof node node-red node-red-pi sscma-node 2>/dev/null
}

# Total private-anonymous resident memory (KiB) of the node stack. Prefers
# RssAnon (see the metric note above); falls back to VmRSS per-process only if
# the kernel does not expose RssAnon (pre-4.5 — this BSP is 5.10 and has it).
_node_rss_kb() {
    total=0
    for pid in $(_node_pids); do
        r=$(awk '/^RssAnon:/ {print $2; exit}' "/proc/$pid/status" 2>/dev/null)
        case "$r" in
        '' | *[!0-9]*)
            r=$(awk '/^VmRSS:/ {print $2; exit}' "/proc/$pid/status" 2>/dev/null)
            ;;
        esac
        case "$r" in
        '' | *[!0-9]*) continue ;;
        esac
        total=$((total + r))
    done
    echo "$total"
}

# Ensure only one watchdog instance runs: record our pid (main.sh also writes
# it, but claim it here too so a direct launch is still tracked).
echo "$$" >"$PIDFILE" 2>/dev/null || true

MAX_KB=$(_max_kb)
_log "start interval=${INTERVAL}s threshold=${MAX_KB}KB (metric=RssAnon, 2-strike)"

# Debounce: only kill after STRIKES consecutive over-threshold samples so a
# transient allocation spike (e.g. during startup) cannot trip the guard.
STRIKES=2
over=0

while : ; do
    # Second line of defense: stop if the device left nodered mode.
    mode=$(head -n 1 "$MODE_FILE" 2>/dev/null | tr -d ' \t\r\n')
    if [ "$mode" != "nodered" ]; then
        _log "mode='$mode' (not nodered) — exiting"
        break
    fi

    total=$(_node_rss_kb)
    if [ "$total" -gt "$MAX_KB" ] 2>/dev/null; then
        over=$((over + 1))
        _log "node stack RssAnon ${total}KB > ${MAX_KB}KB (strike ${over}/${STRIKES})"
        if [ "$over" -ge "$STRIKES" ]; then
            _log "sustained over threshold — killing node stack (serviced will relaunch)"
            killall -9 node-red node-red-pi node sscma-node 2>/dev/null || true
            over=0
        fi
    else
        over=0
    fi

    sleep "$INTERVAL"
done

# Only clear the pid file if it still points at us (avoid clobbering a newer
# instance that main.sh just started).
if [ -f "$PIDFILE" ] && [ "$(cat "$PIDFILE" 2>/dev/null)" = "$$" ]; then
    rm -f "$PIDFILE" 2>/dev/null || true
fi
exit 0
