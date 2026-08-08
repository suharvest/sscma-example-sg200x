#!/bin/sh
set -eu

# Independent CC BY 4.0 external test set (100 clips + temporal labels).
# Source: https://doi.org/10.5281/zenodo.11620083
out=${1:-/tmp/realbiomfall}
mkdir -p "$out"

download() {
    name=$1
    md5_expected=$2
    if [ ! -f "$out/$name" ] || [ "$(md5 -q "$out/$name")" != "$md5_expected" ]; then
        curl -fL --retry 5 --retry-delay 2 \
            -o "$out/$name.part" "https://zenodo.org/records/11620083/files/$name"
        mv "$out/$name.part" "$out/$name"
    fi
    [ "$(md5 -q "$out/$name")" = "$md5_expected" ] || {
        echo "checksum mismatch: $name" >&2
        exit 1
    }
}

download video_clips-trimmed_cropped_padded_resized-100.zip 1168284537004b7937ec552015461719
download labels-100.zip 65a7c3b8346e9cff8497b5bdd5c80372
unzip -oq "$out/video_clips-trimmed_cropped_padded_resized-100.zip" -d "$out"
unzip -oq "$out/labels-100.zip" -d "$out"
echo "RealBiomFall ready in $out"
