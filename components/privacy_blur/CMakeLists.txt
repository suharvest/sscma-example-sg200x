# Privacy masking on the encoded stream: config parsing, Kalman tracking and
# the three CV181x region backends.
#
# `rgn` is the SDK's region library, which owns CVI_RGN_*. It is private
# because nothing in privacy_blur.h mentions a CVI type -- an application asks
# for boxes to be hidden and never learns how.
#
# sscma-micro is required rather than private: ma_img_t and the MA_LOG macros
# appear in the public header, so anything including it needs those headers
# anyway.
set(PRIVACY_BLUR_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/privacy_blur.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/privacy_blur_config.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/mosaic_lut.cpp"
)

component_register(
    COMPONENT_NAME privacy_blur
    SRCS ${PRIVACY_BLUR_SRCS}
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src"
    PRIVATE_REQUIREDS rgn pthread
)
