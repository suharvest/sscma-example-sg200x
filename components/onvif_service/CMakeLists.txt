# Device-level ONVIF: WS-Discovery plus the Device and Media2 SOAP services.
#
# No SOAP toolkit: gSOAP's generated stubs are GPLv2 against this Apache-2.0
# repository, so the XML is templated by hand (docs/onvif-implementation-spec.md
# 14.4). WS-Discovery needs nothing at all -- it is pure UDP multicast.
#
# The SOAP half does need an HTTP listener, and it reuses libwebsockets rather
# than hand-rolling one (spec 14.9, decided in favour of B once lws became the
# only HTTP backend in the tree). Since lws is statically linked into every
# binary already this adds no new dependency.
#
# rtsp_server is required because GetProfiles and GetStreamUri must answer with
# the port and session names actually in use. Asking the component that owns
# them beats keeping a second copy here -- keeping a second copy is exactly how
# the ":554" string stayed wrong across eight applications.
set(ONVIF_SERVICE_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/onvif_discovery.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/onvif_soap.cpp"
)

component_register(
    COMPONENT_NAME onvif_service
    SRCS ${ONVIF_SERVICE_SRCS}
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE_REQUIREDS libwebsockets rtsp_server pthread
)
