#ifndef _LOCAL_ALERT_H_
#define _LOCAL_ALERT_H_

#include <string>

namespace facemesh_reader {

// Edge-side local alert (B-plan: works without network).
// Current impl: stdout banner + flush. Future: GPIO buzzer / LED, ALSA beep.
void fireLocalAlert(const std::string& reason);

// Optional: signal alert ended (cooldown elapsed). Currently logs only.
void clearLocalAlert();

}  // namespace facemesh_reader

#endif  // _LOCAL_ALERT_H_
