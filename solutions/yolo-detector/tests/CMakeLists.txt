cmake_minimum_required(VERSION 3.10)
project(yolo-detector-host-tests LANGUAGES CXX)

# Host-only: no SSCMA, CVI or camera. Builds the inference control unit and the
# HTTP server and exercises them on the development machine.
#
#   cmake -S tests -B tests/build && cmake --build tests/build && ctest --test-dir tests/build

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

enable_testing()

add_executable(inference_control_test
    ${CMAKE_CURRENT_LIST_DIR}/inference_control_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../main/inference_control.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../main/http_control.cpp
)
target_include_directories(inference_control_test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../main
    ${CMAKE_CURRENT_LIST_DIR}/../../../components/json)
target_compile_options(inference_control_test PRIVATE -Wall -Wextra)
target_link_libraries(inference_control_test PRIVATE Threads::Threads)

add_test(NAME inference_control COMMAND inference_control_test)
