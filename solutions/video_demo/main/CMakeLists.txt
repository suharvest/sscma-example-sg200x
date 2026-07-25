link_directories("${CMAKE_CURRENT_LIST_DIR}")

component_register(
    COMPONENT_NAME main
    SRCS "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/"
    PRIVATE_REQUIREDS rtsp_server sophgo
)
