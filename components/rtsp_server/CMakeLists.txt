# rtsp_server.cpp is the only file that knows which streaming library is in
# use. Replacing cvi_rtsp (needed for ONVIF's metadata track, and to escape the
# CVE-carrying live555 2020.07.21 bundled inside it) means rewriting that one
# file; nothing above include/rtsp_server.h changes.
#
# The live555 headers are PRIVATE: only this component reaches through
# CVI_RTSP_CTX to the RTSPServer underneath, to install the auth database that
# cvi_rtsp's own C API does not expose.
set(RTSP_SERVER_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/rtsp_server.cpp"
)

component_register(
    COMPONENT_NAME rtsp_server
    SRCS ${RTSP_SERVER_SRCS}
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE_INCLUDE_DIRS
        "$ENV{SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/include/liveMedia"
        "$ENV{SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/include/groupsock"
        "$ENV{SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/include/UsageEnvironment"
        "$ENV{SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/include/BasicUsageEnvironment"
    PRIVATE_REQUIREDS cvi_rtsp sophgo pthread
)
