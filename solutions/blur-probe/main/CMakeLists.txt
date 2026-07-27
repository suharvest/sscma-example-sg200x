file(GLOB_RECURSE srcs ${CMAKE_CURRENT_LIST_DIR}/*.c ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

set(incs ${CMAKE_CURRENT_LIST_DIR})

# Deliberately shorter than a real solution's list: no model, no MQTT, no ONVIF.
# Every dependency left out is one less thing that can explain a misplaced mask.
#
# debug_stream is the exception, and it earns its place: the JPEG snapshot it
# serves is encoded from the inference frame and so escapes the hardware mask
# entirely, which is a hole the probe has to be able to reproduce and prove
# closed.
component_register(
    COMPONENT_NAME main
    SRCS ${srcs}
    INCLUDE_DIRS ${incs}
    PRIVATE_REQUIREDS sscma-micro sophgo rtsp_server privacy_blur debug_stream
    REQUIREDS opencv_core opencv_imgcodecs opencv_imgproc
)
