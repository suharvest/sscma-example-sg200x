# Device-level ONVIF: discovery now, Device and Media2 SOAP later.
#
# Zero third-party dependencies, and that is a requirement rather than a
# coincidence: mongoose is GPL-2.0 against this Apache-2.0 repository and
# gSOAP's generated stubs are GPLv2, so the pieces of ONVIF that can be built
# without either are the pieces that can ship today. WS-Discovery is pure UDP,
# and the SOAP services are fixed-shape XML that will be templated the same way.
set(ONVIF_SERVICE_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/onvif_discovery.cpp"
)

component_register(
    COMPONENT_NAME onvif_service
    SRCS ${ONVIF_SERVICE_SRCS}
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE_REQUIREDS pthread
)
