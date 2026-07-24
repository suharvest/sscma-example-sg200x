#ifndef __APP_IPCAM_FV_MONITOR_H__
#define __APP_IPCAM_FV_MONITOR_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Background focus-value (FV) publisher.
 * After the ISP is running (plus a 1 s warm-up), polls raw AF hardware
 * statistics via CVI_ISP_GetFocusStatistics() every 200 ms, sums the
 * h0/h1/v0 contrast metrics over the center third of the AF zone grid,
 * and atomically publishes {"fv":<u64>,"ts":<monotonic_ms>} to
 * /tmp/camera_fv.json (write /tmp/.camera_fv.tmp, then rename).
 * Note: deliberately does NOT use CVI_ISP_AFGetFv() — that call lives in
 * libaf.so and dereferences the AF algorithm context, which is NULL on
 * fixed-focus sensors (OV5647) and crashes with SIGSEGV.
 * On failure: nothing is written, one warning is logged, and the call is
 * retried every 30 s. Stop removes the file. */

#define CAMERA_FV_JSON_PATH "/tmp/camera_fv.json"
#define CAMERA_FV_TMP_PATH "/tmp/.camera_fv.tmp"

/* Start the detached publisher thread for the given VI pipe.
 * Safe to call multiple times (no-op when already running). */
int app_ipcam_FvMonitor_Start(int vi_pipe);

/* Stop the publisher thread and remove the published file. */
void app_ipcam_FvMonitor_Stop(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_IPCAM_FV_MONITOR_H__ */
