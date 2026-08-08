#!/bin/sh
set -eu

# Stream a labeled video tree through the physical reCamera NPU and save one
# JSONL trace per clip. The relative directory layout is preserved; symlinked
# test manifests are followed so external sets need not duplicate video data.
#
# Required environment:
#   RECAMERA_PASSWORD   SSH/sudo password (never written to disk)
# Optional:
#   RECAMERA_HOST       default 192.168.42.1
#   RECAMERA_USER       default recamera

if [ "$#" -ne 2 ]; then
    echo "usage: $0 DATASET_DIR OUTPUT_DIR" >&2
    exit 2
fi
: "${RECAMERA_PASSWORD:?RECAMERA_PASSWORD is required}"

DATASET_DIR=$1
OUTPUT_DIR=$2
RECAMERA_HOST=${RECAMERA_HOST:-192.168.42.1}
RECAMERA_USER=${RECAMERA_USER:-recamera}
REMOTE="$RECAMERA_USER@$RECAMERA_HOST"
ASKPASS_LOCAL=$(dirname "$0")/recamera_sudo_askpass.sh
ASKPASS_REMOTE=/tmp/recamera-fall-eval-askpass

export SSHPASS=$RECAMERA_PASSWORD
sshpass -e scp -q -o ProxyCommand=none -o StrictHostKeyChecking=accept-new \
    "$ASKPASS_LOCAL" "$REMOTE:$ASKPASS_REMOTE"
sshpass -e ssh -o ProxyCommand=none -o StrictHostKeyChecking=accept-new "$REMOTE" \
    "chmod 700 '$ASKPASS_REMOTE'"

mkdir -p "$OUTPUT_DIR"
total=$(find -L "$DATASET_DIR" -type f -name '*.mp4' | wc -l | tr -d ' ')
done_count=0

find -L "$DATASET_DIR" -type f -name '*.mp4' | sort | while IFS= read -r video; do
    rel=${video#"$DATASET_DIR"/}
    trace="$OUTPUT_DIR/${rel%.mp4}.jsonl"
    mkdir -p "$(dirname "$trace")"
    if [ -s "$trace" ] && grep -q '"summary"' "$trace"; then
        done_count=$((done_count + 1))
        echo "skip $done_count/$total $rel"
        continue
    fi

    tmp="$trace.incomplete"
    echo "extract $((done_count + 1))/$total $rel"
    {
        printf '%s\n' "$RECAMERA_PASSWORD"
        ffmpeg -nostdin -v error -i "$video" \
            -vf "fps=15,scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:black" \
            -pix_fmt rgb24 -f rawvideo -
    } | sshpass -e ssh -o ProxyCommand=none -o StrictHostKeyChecking=accept-new "$REMOTE" \
        "IFS= read -r CODEX_SUDO_PASS; export CODEX_SUDO_PASS SUDO_ASKPASS=$ASKPASS_REMOTE; exec sudo -A env LD_LIBRARY_PATH=/mnt/system/lib:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd:/mnt/system/lib/3rd:/lib:/usr/lib /usr/local/bin/fall-detection --offline-rgb /dev/stdin --offline-width 640 --offline-height 640 --offline-fps 15" \
        > "$tmp"
    if ! grep -q '"summary"' "$tmp"; then
        echo "missing summary for $rel" >&2
        exit 1
    fi
    mv "$tmp" "$trace"
    done_count=$((done_count + 1))
done

echo "feature extraction complete: $total clips"
