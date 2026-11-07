cmake_minimum_required(VERSION 3.10)
project(face-recognition-host-tests LANGUAGES CXX)

# Host-only. Deliberately does NOT include the solution's toolchain file or
# cmake/project.cmake: nothing here touches SSCMA, CVI, OpenCV or the camera, so
# these tests build and run on the development machine with no cross toolchain
# and no device.
#
#   cmake -S tests -B tests/build && cmake --build tests/build && ctest --test-dir tests/build

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_executable(liveness_temporal_test
    ${CMAKE_CURRENT_LIST_DIR}/liveness_temporal_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../main/liveness_temporal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../main/liveness_capture_writer.cpp
)
target_include_directories(liveness_temporal_test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../main)
target_compile_options(liveness_temporal_test PRIVATE -Wall -Wextra)

add_test(NAME liveness_temporal COMMAND liveness_temporal_test)
