# libdatachannel 0.24.6 (MPL-2.0): WebRTC data channels + WebSocket client, static, for the
# microduck-pose-qa robot link (main/station/webrtc_link.cpp). Built once by fetch_and_build.sh
# (sources pinned by sha256) into prebuilt/riscv64 (git-ignored), like components/libwebsockets.
# OpenSSL is the SDK's 1.1.1k (libssl.so.1.1 / libcrypto.so.1.1, already linked by the app).
set(LDC_PREBUILT "${CMAKE_CURRENT_LIST_DIR}/prebuilt/riscv64")

if(NOT EXISTS "${LDC_PREBUILT}/lib/libdatachannel-static.a")
    message(FATAL_ERROR
        "libdatachannel is not built yet.\n"
        "  Run: components/libdatachannel/fetch_and_build.sh\n"
        "  (inside recamera-build-base:amd64, with the riscv64 toolchain on PATH and SG200X_SDK_PATH set)")
endif()

add_library(libdatachannel INTERFACE)
# Order matters for static archives: libdatachannel before the dependencies it uses.
target_link_libraries(libdatachannel INTERFACE
    "${LDC_PREBUILT}/lib/libdatachannel-static.a"
    "${LDC_PREBUILT}/lib/libjuice-static.a"
    "${LDC_PREBUILT}/lib/libusrsctp.a"
    "${LDC_PREBUILT}/lib/libsrtp2.a"
    ssl crypto pthread)
target_include_directories(libdatachannel INTERFACE "${LDC_PREBUILT}/include")
target_compile_definitions(libdatachannel INTERFACE RTC_STATIC)
include_directories("${LDC_PREBUILT}/include")
add_definitions(-DRTC_STATIC)

message(STATUS "libdatachannel: ${LDC_PREBUILT}")
