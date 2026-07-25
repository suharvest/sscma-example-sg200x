# debug_stream.cpp holds the streaming policy and is library-agnostic; the
# HTTP/WebSocket library is confined to one ws_transport_<lib>.cpp backend
# (see src/ws_transport.h). To swap libraries, add the sibling backend, list it
# here instead of the mongoose one, and change PRIVATE_REQUIREDS -- nothing
# else in this component changes.
set(DEBUG_STREAM_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/debug_stream.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/ws_transport_mongoose.cpp"
)

component_register(
    COMPONENT_NAME debug_stream
    SRCS ${DEBUG_STREAM_SRCS}
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE_REQUIREDS mongoose sophgo pthread
)
