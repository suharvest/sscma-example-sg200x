file(GLOB DEBUG_STREAM_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

component_register(
    COMPONENT_NAME debug_stream
    SRCS ${DEBUG_STREAM_SRCS}
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE_REQUIREDS mongoose sophgo pthread
)
