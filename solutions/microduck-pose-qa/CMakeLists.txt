cmake_minimum_required(VERSION 3.5.0)

project(microduck-pose-qa VERSION 1.0.12 LANGUAGES C CXX)

get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)

message(STATUS "PROJECT_DIR: ${PROJECT_DIR}")

get_filename_component(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

include(${ROOT_DIR}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake)

message(STATUS "ROOT_DIR: ${ROOT_DIR}")

# Pose model: not in git (3.6 MB binary). Default source is the package staging directory.
set(MICRODUCK_POSE_MODEL_DIR "${ROOT_DIR}/../packages/microduck-pose-qa/userdata/local/models"
    CACHE PATH "Directory holding microduck_pose_17kpt_384_int8.cvimodel and .json")
foreach(_model_file microduck_pose_17kpt_384_int8.cvimodel microduck_pose_17kpt_384_int8.json)
    if(NOT EXISTS "${MICRODUCK_POSE_MODEL_DIR}/${_model_file}")
        message(FATAL_ERROR "Model file missing: ${MICRODUCK_POSE_MODEL_DIR}/${_model_file} (set MICRODUCK_POSE_MODEL_DIR)")
    endif()
    install(FILES "${MICRODUCK_POSE_MODEL_DIR}/${_model_file}" DESTINATION /userdata/local/models)
endforeach()

# Keep the calibrated conf file across upgrades (control/conffiles, see cpack_project_config.cmake).
set(CPACK_PROJECT_CONFIG_FILE "${CMAKE_CURRENT_LIST_DIR}/cpack_project_config.cmake")

include(${ROOT_DIR}/cmake/project.cmake)
