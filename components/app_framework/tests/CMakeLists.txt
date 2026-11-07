# Host unit tests for app_framework. Standalone project -- no SDK, no toolchain:
#
#   cmake -S components/app_framework/tests -B /tmp/af-tests
#   cmake --build /tmp/af-tests -j && ctest --test-dir /tmp/af-tests --output-on-failure
#
# Only the SDK-free sources are compiled here (frame, pipeline, event loop,
# HTTP trigger, stream endpoint selection). camera_hub / app / gpio_trigger need the device and are
# covered by the cross build.
cmake_minimum_required(VERSION 3.10)
project(app_framework_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AF_TEST_SANITIZE "Build tests with ASan + UBSan" ON)

set(AF_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

add_library(af_core STATIC
    ${AF_ROOT}/src/frame.cpp
    ${AF_ROOT}/src/pipeline.cpp
    ${AF_ROOT}/src/event_loop.cpp
    ${AF_ROOT}/src/http_trigger.cpp
)
target_include_directories(af_core PUBLIC ${AF_ROOT}/include ${AF_ROOT}/src)
target_compile_options(af_core PUBLIC -Wall -Wextra -Wpedantic -Werror)
find_package(Threads REQUIRED)
target_link_libraries(af_core PUBLIC Threads::Threads)
if(AF_TEST_SANITIZE)
    target_compile_options(af_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    target_link_options(af_core PUBLIC -fsanitize=address,undefined)
endif()

enable_testing()
foreach(t test_frame test_pipeline test_triggers test_stream_config)
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE af_core)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
