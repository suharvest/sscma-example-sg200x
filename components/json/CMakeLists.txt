# nlohmann/json (MIT), header-only.
#
# Lived inside components/mongoose until the libwebsockets migration, which is
# how the supervisor ended up unable to compile without linking a GPL-2.0-only
# library it no longer uses. Keeping an MIT header in a GPL component's
# directory also blurred a licence boundary that matters here: this one is safe
# to ship, mongoose is not.
#
# An INTERFACE target rather than component_register: with no SRCS the macro
# creates no target, and consumers naming `json` in PRIVATE_REQUIREDS would have
# it handed to the linker as -ljson.
add_library(json INTERFACE)
target_include_directories(json INTERFACE "${CMAKE_CURRENT_LIST_DIR}")
include_directories("${CMAKE_CURRENT_LIST_DIR}")
