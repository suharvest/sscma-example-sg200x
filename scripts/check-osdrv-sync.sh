#!/bin/sh
# Fail if the SDK's copy of the kernel driver sources has drifted from the fork.
#
# Two trees hold the same files for different reasons: sg2002_recamera_emmc/ is
# a 1.1 GB build artifact produced by reCamera-OS CI, and reCamera-OS/osdrv is
# the git checkout we actually commit to. Kernel modules must be built from the
# checkout, so that what ships and what is committed cannot differ -- but
# nothing stops someone editing the SDK copy out of habit, and the result would
# be a .ko whose source is nowhere in version control.
#
# This script does not fix anything. It tells you which tree you edited.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SDK="$ROOT/sg2002_recamera_emmc/osdrv"
FORK="$ROOT/reCamera-OS/osdrv"
FILES="interdrv/rgn/chip/cv181x/rgn.c
interdrv/rgn/chip/cv181x/rgn.h
interdrv/vpss/chip/cv181x/vip_sc.c
interdrv/include/chip/cv181x/uapi/linux/rgn_uapi.h"

[ -d "$FORK" ] || { echo "fork checkout missing: $FORK" >&2; exit 2; }

rc=0
for f in $FILES; do
    if [ ! -f "$SDK/$f" ]; then continue; fi
    if ! diff -q "$FORK/$f" "$SDK/$f" >/dev/null 2>&1; then
        echo "DRIFT: $f"
        echo "       fork: $FORK/$f"
        echo "       sdk : $SDK/$f"
        rc=1
    fi
done
[ $rc -eq 0 ] && echo "osdrv sources in sync"
exit $rc
