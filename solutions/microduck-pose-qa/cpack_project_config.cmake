# Loaded by cpack at packaging time (CPACK_PROJECT_CONFIG_FILE). cmake/package.cmake sets the
# maintainer scripts; this adds control/conffiles so opkg/dpkg keep a calibrated
# /etc/microduck-pose-qa.conf (NOMINAL_STAND_* written by calibrate_stand) across upgrades.
list(APPEND CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_LIST_DIR}/control/conffiles")
