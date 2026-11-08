cmake_minimum_required(VERSION 3.10)
project(aruco-detector-host-tests LANGUAGES CXX)

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

add_executable(camera_model_test
    ${CMAKE_CURRENT_LIST_DIR}/camera_model_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../main/camera_model.cpp
)
target_include_directories(camera_model_test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../main)
target_compile_options(camera_model_test PRIVATE -Wall -Wextra)

add_test(NAME camera_model COMMAND camera_model_test)
