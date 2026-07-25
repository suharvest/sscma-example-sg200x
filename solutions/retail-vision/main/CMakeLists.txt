file(GLOB_RECURSE srcs ${CMAKE_CURRENT_LIST_DIR}/*.c ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

set(incs ${CMAKE_CURRENT_LIST_DIR})

component_register(
    COMPONENT_NAME main
    SRCS ${srcs}
    INCLUDE_DIRS ${incs}
    PRIVATE_REQUIREDS json sscma-micro sophgo cvi_rtsp ha_mqtt debug_stream mongoose
    REQUIREDS opencv_core opencv_imgcodecs opencv_imgproc
)
