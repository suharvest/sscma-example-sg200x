file(GLOB_RECURSE srcs ${CMAKE_CURRENT_LIST_DIR}/*.c ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

# FaceMesh blink evidence reuses two files from the facemesh-reader solution
# verbatim: the landmark model runner and the EAR/MAR maths.
#
# They are listed one by one on purpose. facemesh-reader/main also contains a
# YOLO-Face detector, a drowsiness state machine and a yawn detector; globbing
# that directory would link a SECOND face detector into a process whose binding
# constraint is the model ION pool. The crop/mapping FacemeshPipeline would
# otherwise supply is reimplemented in facemesh_liveness.cpp, which is the only
# reason the pipeline itself is absent from this list.
get_filename_component(FACEMESH_READER_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../../facemesh-reader/main ABSOLUTE)

list(APPEND srcs
    ${FACEMESH_READER_DIR}/facemesh_landmarker.cpp
    ${FACEMESH_READER_DIR}/facial_metrics.cpp
)

set(incs ${CMAKE_CURRENT_LIST_DIR} ${FACEMESH_READER_DIR})

component_register(
    COMPONENT_NAME main
    SRCS ${srcs}
    INCLUDE_DIRS ${incs}
    PRIVATE_REQUIREDS onvif_meta onvif_service sscma-micro sophgo rtsp_server ha_mqtt privacy_blur debug_stream json
    REQUIREDS opencv_core opencv_imgcodecs opencv_imgproc
)
