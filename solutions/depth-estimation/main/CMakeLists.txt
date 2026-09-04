file(GLOB_RECURSE srcs ${CMAKE_CURRENT_LIST_DIR}/*.c ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

set(incs ${CMAKE_CURRENT_LIST_DIR})

# `rgn` is the SDK's region library (CVI_RGN_*), needed for the depth picture-in-
# picture overlay. This application does not link privacy_blur: it draws its own
# single OVERLAY region and two ordinary overlay regions on the same VPSS channel
# may not intersect (rgn.c:340-360), so the two features are mutually exclusive
# here and the manifest declares "privacy_blur": false.
component_register(
    COMPONENT_NAME main
    SRCS ${srcs}
    INCLUDE_DIRS ${incs}
    PRIVATE_REQUIREDS onvif_meta onvif_service sscma-micro sophgo rtsp_server ha_mqtt debug_stream rgn
    REQUIREDS opencv_core opencv_imgcodecs opencv_imgproc
)
