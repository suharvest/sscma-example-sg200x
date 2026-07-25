# debug_stream.cpp holds the streaming policy and is library-agnostic; the
# HTTP/WebSocket library is confined to one ws_transport_<lib>.cpp backend
# (see src/ws_transport.h). Exactly one backend is compiled in.
#
# DEBUG_STREAM_BACKEND selects it: "mongoose" (default, incumbent) or "lws".
# The migration to lws exists because mongoose is GPL-2.0-only against this
# repository's Apache-2.0 licence -- see docs/onvif-implementation-spec.md
# 0.5-B. Both can coexist during the transition because lws uses an lws_
# prefix, so debug_stream can switch independently of supervisor.
#
# opencv is pulled in for the /snapshot.jpg JPEG encode only; the video path
# never decodes a frame, it forwards VENC output verbatim.
set(DEBUG_STREAM_BACKEND "mongoose" CACHE STRING "debug_stream HTTP/WS backend")
set_property(CACHE DEBUG_STREAM_BACKEND PROPERTY STRINGS mongoose lws)

if(DEBUG_STREAM_BACKEND STREQUAL "lws")
    set(DEBUG_STREAM_BACKEND_SRC "${CMAKE_CURRENT_LIST_DIR}/src/ws_transport_lws.cpp")
    set(DEBUG_STREAM_BACKEND_LIB libwebsockets)
else()
    set(DEBUG_STREAM_BACKEND_SRC "${CMAKE_CURRENT_LIST_DIR}/src/ws_transport_mongoose.cpp")
    set(DEBUG_STREAM_BACKEND_LIB mongoose)
endif()
message(STATUS "debug_stream backend: ${DEBUG_STREAM_BACKEND}")

set(DEBUG_STREAM_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/debug_stream.cpp"
    "${DEBUG_STREAM_BACKEND_SRC}"
)

component_register(
    COMPONENT_NAME debug_stream
    SRCS ${DEBUG_STREAM_SRCS}
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE_REQUIREDS ${DEBUG_STREAM_BACKEND_LIB} sophgo pthread
        opencv_core opencv_imgcodecs opencv_imgproc
)
