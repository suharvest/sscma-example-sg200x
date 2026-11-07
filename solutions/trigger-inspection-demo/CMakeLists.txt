cmake_minimum_required(VERSION 3.5.0)

project(trigger-inspection-demo VERSION 0.1.0 LANGUAGES C CXX)

get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)

message(STATUS "PROJECT_DIR: ${PROJECT_DIR}")

get_filename_component(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

include(${ROOT_DIR}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake)

message(STATUS "ROOT_DIR: ${ROOT_DIR}")

include(${ROOT_DIR}/cmake/project.cmake)

set(_opencv_runtime_dir
    "${SG200X_SDK_PATH}/cvi_mpi/modules/isp/cv181x/isp-tool-daemon/isp_daemon_tool/install/lib")
set(_private_runtime_libs
    "${_opencv_runtime_dir}/libopencv_core.so.4.5"
    "${_opencv_runtime_dir}/libopencv_imgproc.so.4.5"
    "${_opencv_runtime_dir}/libopencv_imgcodecs.so.4.5"
    "${SG200X_SDK_PATH}/cvi_mpi/lib/libsns_gc2053.so")
foreach(_runtime_lib IN LISTS _private_runtime_libs)
    if(EXISTS "${_runtime_lib}")
        install(FILES ${_runtime_lib} DESTINATION /usr/lib/trigger-inspection-demo)
    endif()
endforeach()
