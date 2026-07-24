#ifndef CAMERA_CONFIG_H
#define CAMERA_CONFIG_H

// Camera picture orientation configuration (mirror / flip / rotation).
//
// Persisted to /userdata/local/camera.conf as plain KEY=value lines
// (shell-sourceable; values are only 0/1/180, no quoting needed):
//   CAM_MIRROR=0|1
//   CAM_FLIP=0|1
//   CAM_ROTATION=0|180
//
// The active application reads this file at startup, so a change only takes
// effect after the app restarts (supervisor reuses the setHaConfig restart
// path). The file holds no secrets -> written atomically with mode 0644
// (tmp + fsync + rename, same pattern as ha.conf).
class camera_config {
public:
    struct conf {
        bool mirror = false;
        bool flip = false;
        int rotation = 0; // 0 or 180
    };

    static constexpr const char* CONF_FILE = "/userdata/local/camera.conf";

    // Read CONF_FILE. Missing/unreadable/malformed lines -> defaults kept
    // (never an error).
    static conf load();

    // Atomic write of CONF_FILE (0644). Returns false on I/O failure.
    static bool save(const conf& c);
};

#endif // CAMERA_CONFIG_H
