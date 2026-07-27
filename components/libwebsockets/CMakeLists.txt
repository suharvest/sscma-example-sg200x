# libwebsockets (MIT) -- the replacement for mongoose, which is GPL-2.0-only
# and therefore incompatible with this repository's Apache-2.0 licence. See
# docs/onvif-implementation-spec.md section 0.5-B.
#
# A private pinned build, NOT the copy the SDK ships. Using the SDK's was tried
# first and is the reason this comment is long: the SDK contains *three*
# libwebsockets at two different versions,
#
#   sysroot/usr/include              lws 4.0.22 headers  <- first on the include path
#   cvitek_tpu_sdk/include           lws 4.1.7  headers
#   sysroot/usr/lib/libwebsockets.so 4.0.22
#   cvitek_tpu_sdk/lib/libwebsockets.a 4.1.7
#
# and project.cmake puts the sysroot include directory ahead of the tpu_sdk
# one. Linking the 4.1.7 archive therefore compiled against 4.0.22 headers.
# The two configurations enable different LWS_WITH_* options, so the fields of
# lws_context_creation_info sit at different offsets: assignments landed on the
# wrong members and lws_create_context faulted at boot. It presented as a
# kernel register dump, not as a build error.
#
# A private build sidesteps the whole thing by owning both sides, and BEFORE
# below puts it ahead of both SDK copies. It is also smaller than the SDK's
# (190 KB of text against 321 KB, since the SDK build carries HTTP/2, TLS and
# client support that is unused here) and pinned rather than tied to whichever
# SDK happens to be installed.
set(LWS_PREBUILT "${CMAKE_CURRENT_LIST_DIR}/prebuilt")

if(NOT EXISTS "${LWS_PREBUILT}/lib/libwebsockets.a")
    message(FATAL_ERROR
        "libwebsockets is not built yet.\n"
        "  Run: components/libwebsockets/fetch_and_build.sh\n"
        "  (inside the build container, with the riscv64 toolchain on PATH)")
endif()

add_library(libwebsockets INTERFACE)
# Absolute path, not -lwebsockets: two other archives of that name are already
# on the link path and either would resolve.
target_link_libraries(libwebsockets INTERFACE "${LWS_PREBUILT}/lib/libwebsockets.a")
target_include_directories(libwebsockets INTERFACE "${LWS_PREBUILT}/include")

# BEFORE is load-bearing. Without it the SDK's 4.0.22 headers win and the
# result is the silent ABI mismatch described above.
include_directories(BEFORE "${LWS_PREBUILT}/include")

message(STATUS "libwebsockets: private ${LWS_PREBUILT} (ahead of SDK 4.0.22/4.1.7)")
