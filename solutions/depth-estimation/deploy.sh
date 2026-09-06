#!/bin/bash
# deploy.sh - Deploy depth-estimation to a reCamera device
# Handles: build -> stop camera-using services -> install deb -> (optionally) start -> verify
#
# Usage:
#   ./deploy.sh                          # Build + install
#   ./deploy.sh --skip-build             # Install the existing deb only
#   ./deploy.sh --host 10.0.0.1          # Custom host
#   ./deploy.sh --start                  # Also start it, bypassing the console
#
# By default this INSTALLS ONLY and leaves starting to the console (appMgr
# switch). That is not timidity — the console owns which app is active:
#
#   * It records the choice in /userdata/local/apps/state.json, which
#     app_restore reads at boot. Starting the init script behind its back
#     leaves that file stale, so a supervisor restart happily launches a
#     SECOND instance of whatever it still thinks is active — two processes,
#     one debug port, and a live-looking app with no debug stream.
#   * Its stop path waits for the VPSS group to be released before starting
#     anything else instead of sleeping and hoping.
#
# --start replicates the VPSS wait, but it still cannot fix state.json.

set -e

# --- Defaults ---
HOST="${RECAMERA_HOST:-192.168.42.1}"
USER="${RECAMERA_USER:-recamera}"
[ -f ~/.recamera ] && . ~/.recamera
PASS="${RECAMERA_PASS:-}"
SKIP_BUILD=false
MQTT_CHECK=true
DO_START=false

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOLUTION_NAME="depth-estimation"
MODEL_FILE="fastdepth_224_bf16.cvimodel"
# Force password auth: sshpass feeds the password to ssh's prompt, but if ssh
# tries publickey or keyboard-interactive first it spawns ssh-askpass instead
# and sshpass never gets to answer ("ssh_askpass: exec(...): No such file").
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o PreferredAuthentications=password -o PubkeyAuthentication=no"

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)       HOST="$2"; shift 2 ;;
        --user)       USER="$2"; shift 2 ;;
        --password)   PASS="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --start)      DO_START=true; shift ;;
        --no-mqtt)    MQTT_CHECK=false; shift ;;
        -h|--help)
            echo "Usage: $0 [--host IP] [--user USER] [--password PASS] [--skip-build] [--start] [--no-mqtt]"
            echo "  --start  start the app directly instead of leaving it to the console"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log()  { echo "==> $*"; }
warn() { echo "WARN: $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }
ok()   { echo "OK: $*"; }

run_ssh()  { sshpass -p "$PASS" ssh $SSH_OPTS "${USER}@${HOST}" "$@"; }
# -O forces the legacy SCP protocol. Modern scp defaults to the SFTP subsystem,
# where sshpass cannot answer the password prompt on some hosts.
run_scp()  { sshpass -p "$PASS" scp -O $SSH_OPTS "$@"; }
run_sudo() { run_ssh "printf '%s\n' '${PASS}' | sudo -S $*"; }

# --- Pre-flight ---
command -v sshpass >/dev/null || err "sshpass not found. Install: brew install hudochenkov/sshpass/sshpass"
ping -c 1 -t 2 "$HOST" >/dev/null 2>&1 || err "Device $HOST not reachable"
[ -z "$PASS" ] && err "No password set. Create ~/.recamera with: RECAMERA_PASS=yourpassword"

# --- Step 1: Build ---
if [ "$SKIP_BUILD" = false ]; then
    log "Building ${SOLUTION_NAME}..."
    docker exec ubuntu_dev_x86 bash -c "
        export SG200X_SDK_PATH=/workspace/sg2002_recamera_emmc
        export PATH=/workspace/host-tools/gcc/riscv64-linux-musl-x86_64/bin:\$PATH
        cd /workspace/sscma-example-sg200x/solutions/${SOLUTION_NAME}
        rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release . && cmake --build build -j4 && cd build && cpack
    " || err "Build failed"
    ok "Build succeeded"
fi

DEB_FILE=$(ls -t "${SCRIPT_DIR}/build/${SOLUTION_NAME}_"*_riscv64.deb 2>/dev/null | head -1)
[ -f "$DEB_FILE" ] || err "No deb package found in ${SCRIPT_DIR}/build/"
DEB_NAME=$(basename "$DEB_FILE")
log "Package: $DEB_NAME"

# --- Step 2: Check the depth model is on the device ---
# It does not ship inside this deb (it is a model, not code), so a device
# without it would start the app only to have it exit on a failed load.
if ! run_sudo "test -f /userdata/local/models/${MODEL_FILE}"; then
    warn "Depth model missing on device: /userdata/local/models/${MODEL_FILE}
     Copy it there before enabling the app, or point MODEL_PATH at another
     depth cvimodel in /etc/depth-estimation.conf."
else
    ok "Depth model present"
fi

# --- Step 3: Stop everything that owns the camera ---
log "Stopping camera-using services on ${HOST}..."
run_ssh 'for svc in /etc/init.d/[SK]*sscma-node* /etc/init.d/[SK]*node-red* \
               /etc/init.d/[SK]*sscma-supervisor* /etc/init.d/[SK]*yolo*detector* \
               /etc/init.d/[SK]*ppocr* /etc/init.d/[SK]*face-analysis* \
               /etc/init.d/[SK]*detection-blur* /etc/init.d/[SK]*retail-vision* \
               /etc/init.d/[SK]*facemesh* /etc/init.d/[SK]*weather-classifier* \
               /etc/init.d/[SK]*fitness-trainer* /etc/init.d/[SK]*depth-estimation*; do
    [ -x "$svc" ] && "$svc" stop 2>/dev/null || true
done' || warn "Some init scripts not found (OK)"

run_sudo 'killall -q depth-estimation fitness-trainer ppocr-reader face-analysis detection-blur retail-vision facemesh-reader weather-classifier yolo-detector sscma-node 2>/dev/null || true'
sleep 2

# Node-RED is watched by /usr/share/supervisor/scripts/nr_memguard.sh, which
# restarts it behind our back. A revived Node-RED grabs the camera while this
# app is streaming, both contend for VPSS, and the pipeline wedges — recoverable
# only by rebooting. Verified: an app that ran 590 frames cleanly wedged at 153
# once Node-RED came back. Stopping the init script alone is not enough.
# Ask the device what mode it is in rather than grepping a process list: a
# pattern like "node-red" also matches the ssh command carrying it, so the grep
# reports a hit even when nothing is running.
if [ "$(run_sudo 'cat /userdata/local/apps/mode 2>/dev/null' | tr -d '\r\n')" = "nodered" ]; then
    err "The device is in Node-RED mode.
     Gallery apps are stopped and disabled in that mode, and Node-RED is watched
     by nr_memguard.sh, which restarts it seconds after the init script stops
     it. A revived Node-RED takes the camera from this app and wedges VPSS.
     Switch the device to Console mode first: open the console, click
     \"Switch back to Console mode\", then enable the app from its card.
     That path goes through appMgr/switch, which stops Node-RED properly,
     waits for the VPSS group and records the active app in state.json."
fi
ok "Services stopped"

# --- Step 4: Transfer & install ---
log "Uploading ${DEB_NAME}..."
run_scp "$DEB_FILE" "${USER}@${HOST}:/tmp/" || err "SCP failed"
ok "Upload complete"

log "Installing package..."
run_sudo "opkg install --force-reinstall /tmp/${DEB_NAME}" || err "Install failed"
ok "Package installed"

# --- Step 5: Start (opt-in) ---
if [ "$DO_START" = false ]; then
    echo ""
    ok "Installed: ${DEB_NAME} -> ${HOST}"
    echo ""
    echo "   Not started. Open the console and enable \"Monocular Depth Estimation\" —"
    echo "   that goes through appMgr/switch, which records the active app in"
    echo "   state.json and waits for the VPSS group before starting."
    echo "   (./deploy.sh --skip-build --start starts it directly instead.)"
    echo ""
    echo "   RTSP:    rtsp://${HOST}:8554/live0"
    echo "   Preview: ws://${HOST}:8001/  (results: ws://${HOST}:8001/results)"
    echo "   MQTT:    recamera/depth-estimation/results"
    exit 0
fi

# Refuse to race the supervisor. When its init script reports FAIL the service
# is still up and may hand the camera to another app mid-run.
if run_ssh 'ps w | grep -q "[s]upervisor -g"'; then
    warn "supervisor is still running; --start bypasses appMgr/switch and can
     race it. Prefer enabling the app from the console instead."
fi

log "Starting ${SOLUTION_NAME} directly (console not involved)..."
run_sudo "/etc/init.d/K92${SOLUTION_NAME} stop" >/dev/null 2>&1 || true

# Wedge-defense, replicated from the supervisor's sh_stop: wait for the
# camera's VPSS group to be handed back before starting the next owner.
# Starting while Grp(0) is still alive yields "Grp(0) is occupied" and then an
# endless "get chn frame fail". Parses the GRP ATTR table of /proc/cvitek/vpss,
# whose data rows start with '#'; with no owner the table has no data rows.
log "Waiting for VPSS Grp(0) to be released..."
if run_sudo "sh -c '
i=0
while [ \$i -lt 25 ]; do
    if ! awk \"/VPSS GRP ATTR/{a=1;next} /-----/{if(a)exit} a&&/^#/{print;exit}\" /proc/cvitek/vpss 2>/dev/null | grep -q .; then
        exit 0
    fi
    sleep 0.2
    i=\$((i+1))
done
exit 1'"; then
    ok "VPSS Grp(0) released"
else
    err "VPSS Grp(0) still held after 5s.
     Something still owns the camera, or the driver is already wedged. Starting
     on top of that is what produces 'Grp(0) is occupied' and then an endless
     'get chn frame fail' — and that state survives restarting the app, so it
     costs a reboot to clear. Check for a lingering app process, or reboot the
     camera, then retry."
fi
sleep 1

run_sudo "/etc/init.d/K92${SOLUTION_NAME} start" || err "Service start failed"
ok "Service started"

# --- Step 6: Verify ---
sleep 5
# Both need root: the pidfile under /var/run and the logfile are root-owned,
# and reading them as the login user reports a running service as stopped.
log "Checking service status..."
run_sudo "/etc/init.d/K92${SOLUTION_NAME} status" || warn "Status check failed"

log "Recent log lines:"
run_sudo "tail -n 25 /var/log/${SOLUTION_NAME}.log" || warn "No log yet"

if [ "$MQTT_CHECK" = true ]; then
    log "Capturing MQTT output (10s, max 3 messages)..."
    run_ssh "timeout 10 mosquitto_sub -h localhost -t 'recamera/${SOLUTION_NAME}/results' -C 3" || warn "MQTT check timed out"
fi

echo ""
ok "Deploy complete: ${DEB_NAME} -> ${HOST}"
echo "   NOTE: started directly — state.json still reflects the console's idea"
echo "         of the active app. Switch once in the console to resync."
echo "   RTSP:    rtsp://${HOST}:8554/live0"
echo "   Preview: ws://${HOST}:8001/  (results: ws://${HOST}:8001/results)"
