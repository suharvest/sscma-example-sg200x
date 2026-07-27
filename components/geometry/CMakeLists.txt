# Normalised box types that carry their own coordinate semantics (centre vs
# corner, inference frame vs video stream). Depends on nothing: no SDK types,
# no logging, no allocation beyond the vector it returns -- so any component or
# application can take it on without inheriting a dependency tree.
component_register(
    COMPONENT_NAME geometry
    SRCS "${CMAKE_CURRENT_LIST_DIR}/src/norm_box.cpp"
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
)
