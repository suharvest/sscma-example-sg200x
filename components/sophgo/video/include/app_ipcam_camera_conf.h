#ifndef __APP_IPCAM_CAMERA_CONF_H__
#define __APP_IPCAM_CAMERA_CONF_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Device-level camera orientation configuration.
 * Parsed from a KEY=VALUE file (default: /userdata/local/camera.conf).
 * Recognized keys (whitelist):
 *   CAM_MIRROR   = 0|1
 *   CAM_FLIP     = 0|1
 *   CAM_ROTATION = 0|180
 * Missing file, unknown keys or invalid values fall back to 0 (never fails).
 */

#define CAMERA_CONF_PATH "/userdata/local/camera.conf"

typedef struct {
    int mirror;   /* CAM_MIRROR: 0|1 */
    int flip;     /* CAM_FLIP: 0|1 */
    int rotation; /* CAM_ROTATION: 0|180 */
} camera_conf_t;

/* Parse conf file into *conf. Always fills defaults (all 0) first;
 * returns 0 if the file was found and read, -1 if it was absent/unreadable
 * (conf still holds valid defaults in that case). */
int app_ipcam_CameraConf_Load(const char* path, camera_conf_t* conf);

#ifdef __cplusplus
}
#endif

#endif /* __APP_IPCAM_CAMERA_CONF_H__ */
