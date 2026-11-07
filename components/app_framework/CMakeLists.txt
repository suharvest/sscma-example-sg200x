# app_framework: camera / streaming bring-up, per-frame pipeline and result
# fan-out shared by the gallery applications (see include/af/app.h).
#
# Sources are listed explicitly so tests/ (host-only, built by its own
# CMakeLists) never ends up in the device library.
set(APP_FRAMEWORK_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/frame.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/pipeline.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/event_loop.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/gpio_trigger.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/http_trigger.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/camera_hub.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/app.cpp"
)

component_register(
    COMPONENT_NAME app_framework
    SRCS ${APP_FRAMEWORK_SRCS}
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src"
    REQUIREDS sscma-micro sophgo
    PRIVATE_REQUIREDS rtsp_server debug_stream ha_mqtt onvif_service pthread
)
