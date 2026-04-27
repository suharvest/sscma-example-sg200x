#include "local_alert.h"

#include <cstdio>

namespace facemesh_reader {

void fireLocalAlert(const std::string& reason) {
    // TODO(hw): wire to GPIO buzzer / status LED.
    std::fprintf(stdout,
        "\n*** DROWSINESS ALERT *** %s\n", reason.c_str());
    std::fflush(stdout);
}

void clearLocalAlert() {
    std::fprintf(stdout, "[alert] cleared\n");
    std::fflush(stdout);
}

}  // namespace facemesh_reader
