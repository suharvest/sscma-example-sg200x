#!/bin/sh
# Fetch and cross-compile libwebsockets for reCamera (riscv64-linux-musl).
#
# Why a script and not vendored sources: the lws tree is ~68 MB with a layered
# roles/plat/tls structure, unlike mongoose which is one file. Vendoring it
# would dominate the repository; building it at CMake configure time would
# require network access during every build. So it is fetched and built once,
# deliberately, and the result is cached in prebuilt/ (git-ignored) the same
# way the SDK and the toolchain live outside the tree.
#
# Run inside the build container:
#   docker exec ubuntu_dev_x86 sh -c \
#     'export SG200X_SDK_PATH=/workspace/sg2002_recamera_emmc; \
#      export PATH=/workspace/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH; \
#      /workspace/sscma-example-sg200x/components/libwebsockets/fetch_and_build.sh'
#
# Everything below is the recipe validated in docs/onvif-implementation-spec.md
# section 14.8: zero warnings, and the two build traps that are not obvious.

set -e

LWS_VERSION="v4.5.8"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${LWS_WORK_DIR:-/tmp/lws-build}"
PREBUILT="$HERE/prebuilt"

TC_PREFIX="${CROSS_COMPILE:-riscv64-unknown-linux-musl-}"
if ! command -v "${TC_PREFIX}gcc" >/dev/null 2>&1; then
    echo "error: ${TC_PREFIX}gcc not on PATH." >&2
    echo "       Add host-tools/gcc/riscv64-linux-musl-x86_64/bin to PATH." >&2
    exit 1
fi

mkdir -p "$WORK"
cd "$WORK"

if [ ! -d libwebsockets ]; then
    echo "==> fetching libwebsockets $LWS_VERSION"
    git clone --depth 1 -b "$LWS_VERSION" https://github.com/warmcat/libwebsockets.git
fi

cat > "$WORK/tc-riscv64-musl.cmake" <<EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_C_COMPILER   ${TC_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TC_PREFIX}g++)
set(CMAKE_AR           ${TC_PREFIX}ar     CACHE FILEPATH "")
set(CMAKE_RANLIB       ${TC_PREFIX}ranlib CACHE FILEPATH "")
set(CMAKE_STRIP        ${TC_PREFIX}strip  CACHE FILEPATH "")
set(CMAKE_C_FLAGS_INIT "-march=rv64imafdcv0p7xthead -mcpu=c906fdv -mabi=lp64d -O2")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF

rm -rf "$WORK/build"
mkdir -p "$WORK/build"
cd "$WORK/build"

# Trap 1: LWS_WITH_JPEG defaults ON and fails under GCC 10.2 because lws adds
#         -Werror and 10.2 misreports jpeg.c:1293 as maybe-uninitialized. The
#         whole 4.5 display stack (JPEG/UPNG/DLO/LHP) is useless here anyway.
# Trap 2: LWS_LOG_TAG_LIFECYCLE=OFF trips an upstream bug at logs.c:171, and
#         -DCMAKE_C_FLAGS=-Wno-error does NOT help: lws appends -Werror after
#         the user's flags. It must stay ON.
# Logs are kept (no LWS_WITH_NO_LOGS): 8 KB for the ability to diagnose a
# device in the field is a trade worth making.
cmake "$WORK/libwebsockets" \
    -DCMAKE_TOOLCHAIN_FILE="$WORK/tc-riscv64-musl.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREBUILT" \
    -DLWS_WITH_SSL=OFF \
    -DLWS_WITHOUT_TESTAPPS=ON -DLWS_WITH_MINIMAL_EXAMPLES=OFF \
    -DLWS_WITH_HTTP2=OFF \
    -DLWS_ROLE_MQTT=OFF -DLWS_ROLE_DBUS=OFF \
    -DLWS_ROLE_RAW_PROXY=OFF -DLWS_ROLE_RAW_FILE=OFF \
    -DLWS_WITH_ZLIB=OFF -DLWS_WITH_ZIP_FOPS=OFF \
    -DLWS_WITH_HTTP_STREAM_COMPRESSION=OFF \
    -DLWS_WITHOUT_CLIENT=ON \
    -DLWS_WITH_LIBUV=OFF -DLWS_WITH_LIBEVENT=OFF -DLWS_WITH_LIBEV=OFF \
    -DLWS_WITH_GLIB=OFF -DLWS_WITH_SDEVENT=OFF -DLWS_WITH_ULOOP=OFF \
    -DLWS_WITH_STATIC=ON -DLWS_WITH_SHARED=OFF \
    -DLWS_WITH_JPEG=OFF -DLWS_WITH_UPNG=OFF \
    -DLWS_WITH_DLO=OFF -DLWS_WITH_LHP=OFF \
    -DLWS_IPV6=OFF -DLWS_WITH_PLUGINS=OFF \
    -DLWS_WITH_SYS_ASYNC_DNS=OFF -DLWS_WITH_SYS_NTPCLIENT=OFF \
    -DLWS_WITH_SYS_DHCP_CLIENT=OFF \
    -DLWS_WITH_CONMON=OFF -DLWS_WITH_SYS_STATE=OFF \
    -DLWS_WITH_SYS_SMD=OFF -DLWS_WITH_SYS_METRICS=OFF \
    -DLWS_WITH_SECURE_STREAMS=OFF -DLWS_WITH_NETLINK=OFF -DLWS_WITH_UDP=OFF \
    -DLWS_WITH_LEJP=OFF -DLWS_WITH_LEJP_CONF=OFF -DLWS_WITH_LWSAC=OFF \
    -DLWS_WITH_STRUCT_JSON=OFF -DLWS_WITH_CBOR=OFF -DLWS_WITH_COSE=OFF \
    -DLWS_WITH_JOSE=OFF -DLWS_WITH_GENCRYPTO=OFF \
    -DLWS_WITH_CACHE_NSCOOKIEJAR=OFF \
    -DLWS_WITH_HTTP_UNCOMMON_HEADERS=OFF -DLWS_WITH_CUSTOM_HEADERS=OFF \
    -DLWS_WITH_ACCESS_LOG=OFF -DLWS_WITH_RANGES=OFF \
    -DLWS_WITH_CGI=OFF -DLWS_WITH_SPAWN=OFF -DLWS_WITH_PEER_LIMITS=OFF \
    -DLWS_WITH_SYS_FAULT_INJECTION=OFF -DLWS_WITHOUT_EXTENSIONS=ON \
    -DLWS_WITH_HTTP_BASIC_AUTH=OFF -DLWS_WITH_HTTP_DIGEST_AUTH=OFF \
    -DLWS_WITH_HTTP_PROXY=OFF \
    -DLWS_WITH_THREADPOOL=OFF -DLWS_WITH_DIR=OFF -DLWS_WITH_FTS=OFF \
    -DLWS_WITH_DISKCACHE=OFF -DLWS_WITH_LWS_DSH=OFF \
    -DLWS_WITH_SUL_DEBUGGING=OFF -DLWS_WITH_TLS_SESSIONS=OFF \
    -DLWS_LOG_TAG_LIFECYCLE=ON

make -j"$(nproc)"
rm -rf "$PREBUILT"
make install

echo
echo "==> installed to $PREBUILT"
"${TC_PREFIX}size" -t "$PREBUILT"/lib/libwebsockets.a 2>/dev/null | tail -1 || true
ls -l "$PREBUILT/lib/" 2>/dev/null
