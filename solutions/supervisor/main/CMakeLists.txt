# See http_server.h: mongoose (GPL-2.0-only) is being replaced by
# libwebsockets (MIT). Default stays mongoose until the lws path is verified on
# hardware; -DSUPERVISOR_HTTP_BACKEND=lws selects the new one.
set(SUPERVISOR_HTTP_BACKEND "mongoose" CACHE STRING "supervisor HTTP backend")
set_property(CACHE SUPERVISOR_HTTP_BACKEND PROPERTY STRINGS mongoose lws)

if(SUPERVISOR_HTTP_BACKEND STREQUAL "lws")
    add_definitions(-DSUPERVISOR_HTTP_BACKEND_LWS)
    set(SUPERVISOR_HTTP_LIB libwebsockets)
else()
    set(SUPERVISOR_HTTP_LIB mongoose)
endif()
message(STATUS "supervisor HTTP backend: ${SUPERVISOR_HTTP_BACKEND}")

file(GLOB_RECURSE srcs
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

set(incs ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

set(OPTION_TLS MG_TLS_NONE CACHE INTERNAL "")

component_register(
    COMPONENT_NAME main
    SRCS ${srcs}
    INCLUDE_DIRS ${incs}
    PRIVATE_REQUIREDS json ${SUPERVISOR_HTTP_LIB} mosquitto ssl crypto cares
)
