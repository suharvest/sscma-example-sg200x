#!/bin/sh
# Fetch and build libdatachannel (MPL-2.0) as static archives, for the WebRTC robot link of
# microduck-pose-qa (main/station/webrtc_link.cpp). Same pattern as components/libwebsockets:
# fetched and built once, deliberately; the result lives in prebuilt/<target>/ (git-ignored).
#
#   TARGET=riscv64 (default): riscv64-linux-musl with the SDK toolchain and the SDK's OpenSSL
#     1.1.1k (the app already links libssl.so.1.1 / libcrypto.so.1.1 through mosquitto). Inside
#     recamera-build-base:amd64:
#       docker run --rm --platform linux/amd64 -v /home/harvest/project/recamera:/workspace \
#         recamera-build-base:amd64 sh -c 'export SG200X_SDK_PATH=/workspace/sg2002_recamera_emmc; \
#         export PATH=/workspace/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH; \
#         /workspace/sscma-example-sg200x/components/libdatachannel/fetch_and_build.sh'
#   TARGET=host: native build with the system OpenSSL (host tests, mdqa-hosttest:m2 image).
#
# GitHub source archives carry no submodules, so libdatachannel and each dependency it builds
# (the submodule commits of the v0.24.6 tag) are fetched separately and pinned by sha256.
# deps/json is only used by the examples and is not fetched.
set -e

TARGET="${TARGET:-riscv64}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${LDC_WORK_DIR:-/tmp/ldc-build-$TARGET}"
DL="${LDC_DL_DIR:-$WORK/dl}"
PREBUILT="$HERE/prebuilt/$TARGET"

LDC_VERSION="0.24.6"
# name|url|sha256|directory inside the libdatachannel tree
SOURCES="
libdatachannel|https://github.com/paullouisageneau/libdatachannel/archive/refs/tags/v${LDC_VERSION}.tar.gz|22a66aebebd8dae561d8e5181da96f2fcc203e067c5b2186c07b39af913464c0|.
libjuice|https://github.com/paullouisageneau/libjuice/archive/b89c792e3612faf2f12cf35bcc56857313a06be3.tar.gz|713b22174e8cb2e06538fa9bac2650bda21b83c86f5128813ab4dcf8f5bea9ed|deps/libjuice
usrsctp|https://github.com/paullouisageneau/usrsctp/archive/fec583d54493f879d2ae44a743423bf8a04371ab.tar.gz|e5c114afe73c9a0ec419fab5f5b3f63f3ce57b09b90d06ea85e63dca8aed3e7c|deps/usrsctp
libsrtp|https://github.com/cisco/libsrtp/archive/24b3bf8f19b6f5ab4cd2bcceb4f4064efca86fd5.tar.gz|063478e368d7cd13d04a908d152a46f90bfa728c4b324be6d413b0b92728207c|deps/libsrtp
plog|https://github.com/SergiusTheBest/plog/archive/94899e0b926ac1b0f4750bfbd495167b4a6ae9ef.tar.gz|92a08bce559b5f28aa88d3fd9071567414b9f43a83fe2a05a6dd14f1da536072|deps/plog
"

mkdir -p "$WORK" "$DL"
SRC="$WORK/libdatachannel-$LDC_VERSION"
rm -rf "$SRC"
mkdir -p "$SRC"
echo "$SOURCES" | while IFS='|' read -r name url sha dir; do
    [ -n "$name" ] || continue
    f="$DL/$name-$(echo "$url" | sed 's#.*/##')"
    if [ ! -f "$f" ]; then
        echo "==> fetching $name"
        curl -sSfL -o "$f.part" "$url"
        mv "$f.part" "$f"
    fi
    echo "$sha  $f" | sha256sum -c - >/dev/null || { echo "error: sha256 mismatch for $f" >&2; exit 1; }
    echo "==> $name sha256 ok ($sha)"
    mkdir -p "$SRC/$dir"
    tar xzf "$f" -C "$SRC/$dir" --strip-components=1
done

# Size: no media handlers beyond SRTP are needed, but media stays enabled (NO_MEDIA=OFF) because
# mediad's offer carries a video m-line and the DTLS-SRTP handshake must match what a browser /
# aiortc does on that transport. Static archives, -ffunction-sections so the app link can drop
# what it does not use.
COMMON="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DNO_EXAMPLES=ON -DNO_TESTS=ON \
 -DNO_WEBSOCKET=OFF -DNO_MEDIA=OFF -DUSE_GNUTLS=OFF -DUSE_MBEDTLS=OFF -DUSE_NICE=OFF \
 -DPREFER_SYSTEM_LIB=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$PREBUILT"
OPT="-O2 -ffunction-sections -fdata-sections"

BUILD="$WORK/build"
rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"
if [ "$TARGET" = "riscv64" ]; then
    TC_PREFIX="${CROSS_COMPILE:-riscv64-unknown-linux-musl-}"
    command -v "${TC_PREFIX}gcc" >/dev/null 2>&1 || { echo "error: ${TC_PREFIX}gcc not on PATH" >&2; exit 1; }
    : "${SG200X_SDK_PATH:?set SG200X_SDK_PATH (the SDK holds OpenSSL)}"
    SYSROOT="$SG200X_SDK_PATH/buildroot-2021.05/output/cvitek_CV181X_musl_riscv64/host/riscv64-buildroot-linux-musl/sysroot"
    ARCHF="-march=rv64gcv0p7_zfh_xthead -mcpu=c906fdv -mabi=lp64d"
    cat > "$WORK/tc.cmake" <<TC
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_C_COMPILER   ${TC_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TC_PREFIX}g++)
set(CMAKE_AR           ${TC_PREFIX}ar     CACHE FILEPATH "")
set(CMAKE_RANLIB       ${TC_PREFIX}ranlib CACHE FILEPATH "")
set(CMAKE_STRIP        ${TC_PREFIX}strip  CACHE FILEPATH "")
set(CMAKE_C_FLAGS_INIT   "$ARCHF $OPT")
set(CMAKE_CXX_FLAGS_INIT "$ARCHF $OPT")
set(CMAKE_FIND_ROOT_PATH $SYSROOT)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
TC
    # shellcheck disable=SC2086
    cmake "$SRC" -DCMAKE_TOOLCHAIN_FILE="$WORK/tc.cmake" $COMMON \
        -DOPENSSL_ROOT_DIR="$SYSROOT/usr" -DOPENSSL_INCLUDE_DIR="$SYSROOT/usr/include" \
        -DOPENSSL_SSL_LIBRARY="$SYSROOT/usr/lib/libssl.so" -DOPENSSL_CRYPTO_LIBRARY="$SYSROOT/usr/lib/libcrypto.so"
else
    # shellcheck disable=SC2086
    cmake "$SRC" $COMMON -DCMAKE_C_FLAGS="$OPT" -DCMAKE_CXX_FLAGS="$OPT"
fi
make -j"$(nproc)" datachannel-static
rm -rf "$PREBUILT"
mkdir -p "$PREBUILT/lib" "$PREBUILT/include"
cp -r "$SRC/include/rtc" "$PREBUILT/include/"
# libdatachannel.a and the static dependency archives it was built with
for a in $(find "$BUILD" -name '*.a'); do cp "$a" "$PREBUILT/lib/"; done
{
    echo "libdatachannel $LDC_VERSION ($TARGET)"
    echo "$SOURCES" | awk -F'|' 'NF==4 {print $1, $3}'
    echo "cmake: $COMMON"
    echo "cflags: ${ARCHF:-} $OPT"
} > "$PREBUILT/BUILDINFO"
echo
echo "==> installed to $PREBUILT"
ls -l "$PREBUILT/lib"
cat "$PREBUILT/BUILDINFO"
