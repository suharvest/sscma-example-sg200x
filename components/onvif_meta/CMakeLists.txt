# Pure data model, config parsing and serialisers: no transport, no ONVIF
# service, no third party dependency. The same structures will feed the XML
# serialiser when the RTSP metadata track lands, so keep MQTT and RTSP out of
# this component.
set(ONVIF_META_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/onvif_meta.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/onvif_meta_config.cpp"
)

component_register(
    COMPONENT_NAME onvif_meta
    SRCS ${ONVIF_META_SRCS}
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
)
