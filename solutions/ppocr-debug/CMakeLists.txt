cmake_minimum_required(VERSION 3.5.0)

project(ppocr-debug VERSION 0.1.0 LANGUAGES C CXX)

set(PROJECT_DESCRIPTION "PP-OCRv3 Debug Tool for ReCamera")

get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

include(${ROOT_DIR}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake)
include(${ROOT_DIR}/cmake/project.cmake)
