#ifndef __APP_IPCAM_CONF_WATCHER_H__
#define __APP_IPCAM_CONF_WATCHER_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Background watcher for /userdata/local/camera.conf (CAMERA_CONF_PATH).
 *
 * Polls the conf file's mtime once per second after the video pipeline is
 * up. When the file changes (including appearing or being deleted), it
 * reloads the conf, computes the new effective orientation
 * (mirror XOR rot180, flip XOR rot180) and applies only the DELTA against
 * the last applied effective values directly on the running VI channel via
 * CVI_VI_GetChnFlipMirror / CVI_VI_SetChnFlipMirror. Applying the delta (a
 * conditional toggle of the live hardware state) keeps both the sensor's
 * base orientation and any application-level setVideoMirror()/setVideoFlip()
 * preference intact, exactly mirroring the XOR composition the startup path
 * (applyCameraConf() + app_ipcam_Vi_Chn_Start()) uses.
 *
 * A missing conf file is treated as all-zero defaults, so deleting the file
 * reverts the picture to the default orientation at runtime.
 *
 * On a CVI API failure the watcher warns once and retries every 30 s.
 */

/* Start the watcher thread (detached). vi_pipe/vi_chn must match the values
 * used by the VI init path. applied_mirror_eff/applied_flip_eff are the
 * effective conf values that the startup path already folded into VI
 * (i.e. what applyCameraConf() computed) — the watcher's baseline. */
int app_ipcam_ConfWatcher_Start(int vi_pipe, int vi_chn, int applied_mirror_eff, int applied_flip_eff);

/* Stop the watcher thread (blocks briefly until it drains). */
void app_ipcam_ConfWatcher_Stop(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_IPCAM_CONF_WATCHER_H__ */
