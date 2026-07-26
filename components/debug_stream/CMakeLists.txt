# debug_stream.cpp holds the streaming policy and is library-agnostic; the
# HTTP/WebSocket library is confined to one ws_transport_<lib>.cpp backend
# (see src/ws_transport.h). Keeping that seam is worthwhile even now that only
# one backend exists -- it is what made replacing mongoose a matter of adding
# a file rather than editing the streaming logic.
#
# opencv is pulled in for the /snapshot.jpg JPEG encode only; the video path
# never decodes a frame, it forwards VENC output verbatim.
set(DEBUG_STREAM_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/debug_stream.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/ws_transport_lws.cpp"
)

component_register(
    COMPONENT_NAME debug_stream
    SRCS ${DEBUG_STREAM_SRCS}
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE_REQUIREDS libwebsockets sophgo pthread
        opencv_core opencv_imgcodecs opencv_imgproc
)
