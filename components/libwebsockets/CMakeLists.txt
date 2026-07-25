# libwebsockets (MIT) -- the replacement for mongoose, which is GPL-2.0-only
# and therefore incompatible with this repository's Apache-2.0 licence. See
# docs/onvif-implementation-spec.md section 0.5-B for why this migration
# exists, and 14.8 for the pinned build recipe.
#
# Artifacts come from fetch_and_build.sh rather than from vendored sources or a
# configure-time download; run it once per checkout. Failing loudly here beats
# a link error thirty seconds later with no explanation.
set(LWS_PREBUILT "${CMAKE_CURRENT_LIST_DIR}/prebuilt")

if(NOT EXISTS "${LWS_PREBUILT}/lib/libwebsockets.a")
    message(FATAL_ERROR
        "libwebsockets is not built yet.\n"
        "  Run: components/libwebsockets/fetch_and_build.sh\n"
        "  (inside the build container, with the riscv64 toolchain on PATH)")
endif()

component_register(
    COMPONENT_NAME libwebsockets
    INCLUDE_DIRS "${LWS_PREBUILT}/include"
    LIBRARY_DIRS "${LWS_PREBUILT}/lib"
)
