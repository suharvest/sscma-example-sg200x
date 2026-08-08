#!/bin/sh
set -eu

# Download the public GMDCSA-24 v2.1 videos through a GitHub mirror. Existing
# non-empty clips are retained, so interrupted runs resume without re-fetching
# completed files. Usage: ./download_gmdcsa24.sh [output-dir]

OUT=${1:-/tmp/gmdcsa24}
BASE='https://ghproxy.net/https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/raw/v2.1'
JOBS=${JOBS:-4}
mkdir -p "$OUT"

emit_jobs() {
    subject=1
    while [ "$subject" -le 4 ]; do
        case "$subject" in
            1) adl=16; falls=16 ;;
            2) adl=23; falls=25 ;;
            3) adl=22; falls=21 ;;
            4) adl=20; falls=17 ;;
        esac
        i=1
        while [ "$i" -le "$adl" ]; do
            printf '%s %s %s\n' "$subject" ADL "$i"
            i=$((i + 1))
        done
        i=1
        while [ "$i" -le "$falls" ]; do
            printf '%s %s %s\n' "$subject" Fall "$i"
            i=$((i + 1))
        done
        subject=$((subject + 1))
    done
}

export OUT BASE
emit_jobs | xargs -P "$JOBS" -n 3 sh -c '
    subject=$1
    class=$2
    number=$(printf "%02d" "$3")
    dir="$OUT/subject-$subject/$class"
    dst="$dir/$number.mp4"
    mkdir -p "$dir"
    part="$dst.part"
    if [ -s "$dst" ]; then
        exit 0
    fi
    url="$BASE/Subject%20$subject/$class/$number.mp4"
    echo "download subject-$subject/$class/$number.mp4"
    curl --http1.1 -fsSL --show-error --retry 6 --retry-all-errors \
        --retry-delay 2 --continue-at - "$url" -o "$part"
    mv "$part" "$dst"
' _

subject=1
while [ "$subject" -le 4 ]; do
    for class in ADL Fall; do
        csv="$OUT/subject-$subject/$class.csv"
        if [ ! -s "$csv" ]; then
            curl --http1.1 -fsSL --show-error --retry 6 --retry-all-errors \
                "$BASE/Subject%20$subject/$class.csv" -o "$csv"
        fi
    done
    subject=$((subject + 1))
done

count=$(find "$OUT" -type f -name '*.mp4' | wc -l | tr -d ' ')
if [ "$count" -ne 160 ]; then
    echo "expected 160 MP4 files, found $count" >&2
    exit 1
fi
echo "GMDCSA-24 ready: $count videos in $OUT"
