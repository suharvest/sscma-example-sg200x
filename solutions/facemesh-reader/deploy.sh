#!/bin/bash
# deploy.sh - Deploy facemesh-reader to reCamera device
# Handles: build → upload FaceMesh cvimodel → stop conflicting camera services →
#          install deb → start service → verify
#
# Usage:
#   ./deploy.sh                          # Build + deploy
#   ./deploy.sh --skip-build             # Deploy existing deb only
#   ./deploy.sh --skip-model             # Skip uploading the FaceMesh cvimodel
#   ./deploy.sh --host 10.0.0.1          # Custom host
#   ./deploy.sh --password mypass        # Custom password

set -e

# --- Defaults ---
HOST="${RECAMERA_HOST:-192.168.42.1}"
USER="${RECAMERA_USER:-recamera}"
# Load credentials from ~/.recamera if exists
[ -f ~/.recamera ] && . ~/.recamera
PASS="${RECAMERA_PASS:-}"
SKIP_BUILD=false
SKIP_MODEL=false
MQTT_CHECK=true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOLUTION_NAME="facemesh-reader"
FACEMESH_MODEL_LOCAL="${HOME}/project/recamera/model_conversion/recamera_facemesh/face_landmark_cv181x_bf16.cvimodel"
FACEMESH_MODEL_REMOTE="/userdata/local/models/face_landmark_cv181x_bf16.cvimodel"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)        HOST="$2"; shift 2 ;;
        --user)        USER="$2"; shift 2 ;;
        --password)    PASS="$2"; shift 2 ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --skip-model)  SKIP_MODEL=true; shift ;;
        --no-mqtt)     MQTT_CHECK=false; shift ;;
        -h|--help)
            echo "Usage: $0 [--host IP] [--user USER] [--password PASS] [--skip-build] [--skip-model] [--no-mqtt]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log()  { echo "==> $*"; }
warn() { echo "WARN: $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }
ok()   { echo "OK: $*"; }

# SSH/SCP wrappers
run_ssh() {
    sshpass -p "$PASS" ssh $SSH_OPTS "${USER}@${HOST}" "$@"
}

run_scp() {
    sshpass -p "$PASS" scp $SSH_OPTS "$@"
}

run_sudo() {
    run_ssh "printf '%s\n' '${PASS}' | sudo -S $*"
}

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

# Find deb package
DEB_FILE=$(ls -t "${SCRIPT_DIR}/build/${SOLUTION_NAME}_"*_riscv64.deb 2>/dev/null | head -1)
[ -f "$DEB_FILE" ] || err "No deb package found in ${SCRIPT_DIR}/build/"
DEB_NAME=$(basename "$DEB_FILE")
log "Package: $DEB_NAME"

# --- Step 2: Upload FaceMesh cvimodel ---
if [ "$SKIP_MODEL" = false ]; then
    if [ -f "$FACEMESH_MODEL_LOCAL" ]; then
        log "Uploading FaceMesh cvimodel..."
        run_ssh "mkdir -p /userdata/local/models" || true
        run_scp "$FACEMESH_MODEL_LOCAL" "${USER}@${HOST}:${FACEMESH_MODEL_REMOTE}" || err "Model upload failed"
        ok "FaceMesh model uploaded to ${FACEMESH_MODEL_REMOTE}"
    else
        warn "FaceMesh model not found at ${FACEMESH_MODEL_LOCAL} (skipping upload)"
    fi
fi

# --- Step 3: Stop ALL camera-using services on device ---
log "Stopping all camera-using services on ${HOST}..."

run_ssh 'for svc in /etc/init.d/S*sscma-node* /etc/init.d/K*sscma-node* \
               /etc/init.d/S*node-red* /etc/init.d/K*node-red* \
               /etc/init.d/S*sscma-supervisor* /etc/init.d/K*sscma-supervisor* \
               /etc/init.d/S*yolo*detector* /etc/init.d/K*yolo*detector* \
               /etc/init.d/S*ppocr* /etc/init.d/K*ppocr* \
               /etc/init.d/S*face-analysis* /etc/init.d/K*face-analysis* \
               /etc/init.d/S*facemesh-reader* /etc/init.d/K*facemesh-reader* \
               /etc/init.d/S*detection-blur* /etc/init.d/K*detection-blur* \
               /etc/init.d/S*retail-vision* /etc/init.d/K*retail-vision*; do
    [ -x "$svc" ] && "$svc" stop 2>/dev/null || true
done' || warn "Some init scripts not found (OK)"

run_sudo 'killall -q facemesh-reader face-analysis detection-blur retail-vision ppocr-reader yolo8-detector yolo11-detector yolo26-detector yolo11s-detector sscma-node 2>/dev/null || true'
sleep 2
ok "Services stopped"

# --- Step 4: Transfer & Install ---
log "Uploading ${DEB_NAME}..."
run_scp "$DEB_FILE" "${USER}@${HOST}:/tmp/" || err "SCP failed"
ok "Upload complete"

log "Installing package..."
run_sudo "opkg install --force-reinstall /tmp/${DEB_NAME}" || err "Install failed"
ok "Package installed"

# --- Step 5: Start service ---
log "Starting ${SOLUTION_NAME}..."
run_sudo "/etc/init.d/S92facemesh-reader restart" || err "Service start failed"
ok "Service started"

# --- Step 6: Verify ---
sleep 5

log "Checking service status..."
run_ssh "/etc/init.d/S92facemesh-reader status" || warn "Status check failed"

if [ "$MQTT_CHECK" = true ]; then
    log "Capturing MQTT output (10s, max 3 messages)..."
    run_ssh "timeout 10 mosquitto_sub -h localhost -t 'recamera/facemesh-reader/results' -C 3" || warn "MQTT check timed out"
fi

echo ""
ok "Deploy complete: ${DEB_NAME} -> ${HOST}"
