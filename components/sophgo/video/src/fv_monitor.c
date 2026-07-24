/* Background focus-value (FV) publisher. See app_ipcam_fv_monitor.h. */

#include "app_ipcam_fv_monitor.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

#include "app_ipcam_paramparse.h" /* APP_PROF_LOG_PRINT */

/* CVI_ISP_GetFocusStatistics(VI_PIPE, ISP_AF_STATISTICS_S*): raw ISP
 * hardware AF statistics, exported by libisp.so. Unlike CVI_ISP_AFGetFv
 * (libaf.so), it does NOT go through the AF algorithm library, so it is
 * safe on fixed-focus sensors (e.g. OV5647) where no AF algo is
 * registered and libaf's internal context is NULL. */
#include <cvi_isp.h>

#define FV_POLL_INTERVAL_US (200 * 1000) /* 200 ms */
#define FV_RETRY_BACKOFF_MS (30 * 1000)  /* 30 s after a failure */
#define FV_WARMUP_US (1000 * 1000)       /* let ISP statistics settle */

static volatile int s_fv_run   = 0;
static volatile int s_fv_alive = 0;
static int s_fv_pipe           = 0;

static uint64_t fv_monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)(ts.tv_nsec / 1000000);
}

/* Write buf atomically: tmp file + rename. No heap allocation. */
static void fv_publish(const char* buf, size_t len) {
    int fd = open(CAMERA_FV_TMP_PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return;
    }
    ssize_t wr = write(fd, buf, len);
    close(fd);
    if (wr == (ssize_t)len) {
        rename(CAMERA_FV_TMP_PATH, CAMERA_FV_JSON_PATH);
    } else {
        unlink(CAMERA_FV_TMP_PATH);
    }
}

/* Sum the h0/h1 (horizontal FIR gradient) and v0 (vertical gradient)
 * contrast metrics over the center third of the AF zone grid. Higher
 * value = sharper image. Grid is AF_ZONE_ROW x AF_ZONE_COLUMN (15x17). */
static uint64_t fv_from_af_stats(const ISP_AF_STATISTICS_S* st) {
    uint64_t sum        = 0;
    const int row_start = AF_ZONE_ROW / 3;
    const int row_end   = AF_ZONE_ROW - AF_ZONE_ROW / 3;
    const int col_start = AF_ZONE_COLUMN / 3;
    const int col_end   = AF_ZONE_COLUMN - AF_ZONE_COLUMN / 3;

    for (int r = row_start; r < row_end; r++) {
        for (int c = col_start; c < col_end; c++) {
            const ISP_FOCUS_ZONE_S* z = &st->stFEAFStat.stZoneMetrics[r][c];
            sum += z->u64h0 + z->u64h1 + (uint64_t)z->u32v0;
        }
    }
    return sum;
}

static void* fv_thread(void* arg) {
    (void)arg;
    char buf[96];
    int warned           = 0;
    uint64_t next_try_ms = 0;
    /* ~8 KB; static keeps it off the thread stack (single instance). */
    static ISP_AF_STATISTICS_S s_af_stat;

    prctl(PR_SET_NAME, "FV_MONITOR", 0, 0, 0);
    s_fv_alive = 1;

    /* Give the ISP a moment to produce its first statistics frames. */
    usleep(FV_WARMUP_US);

    while (s_fv_run) {
        uint64_t now = fv_monotonic_ms();
        if (now >= next_try_ms) {
            memset(&s_af_stat, 0, sizeof(s_af_stat));
            CVI_S32 s32Ret = CVI_ISP_GetFocusStatistics((VI_PIPE)s_fv_pipe, &s_af_stat);
            if (s32Ret == CVI_SUCCESS) {
                warned      = 0;
                next_try_ms = 0; /* back to every poll tick */
                uint64_t fv = fv_from_af_stats(&s_af_stat);
                int len     = snprintf(buf, sizeof(buf), "{\"fv\":%llu,\"ts\":%llu}\n", (unsigned long long)fv, (unsigned long long)now);
                if (len > 0 && len < (int)sizeof(buf)) {
                    fv_publish(buf, (size_t)len);
                }
            } else {
                if (!warned) {
                    APP_PROF_LOG_PRINT(LEVEL_WARN, "CVI_ISP_GetFocusStatistics(%d) failed with %#x, retry every %ds\n", s_fv_pipe, s32Ret, FV_RETRY_BACKOFF_MS / 1000);
                    warned = 1;
                }
                next_try_ms = now + FV_RETRY_BACKOFF_MS;
            }
        }
        usleep(FV_POLL_INTERVAL_US);
    }

    unlink(CAMERA_FV_JSON_PATH);
    unlink(CAMERA_FV_TMP_PATH);
    s_fv_alive = 0;
    return NULL;
}

int app_ipcam_FvMonitor_Start(int vi_pipe) {
    if (s_fv_run) {
        return 0; /* already running */
    }
    s_fv_pipe = vi_pipe;
    s_fv_run  = 1;

    pthread_t tid;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    int ret = pthread_create(&tid, &attr, fv_thread, NULL);
    pthread_attr_destroy(&attr);
    if (ret != 0) {
        s_fv_run = 0;
        APP_PROF_LOG_PRINT(LEVEL_WARN, "create fv monitor thread failed (%d)\n", ret);
        return -1;
    }
    APP_PROF_LOG_PRINT(LEVEL_INFO, "fv monitor started on pipe %d -> %s\n", vi_pipe, CAMERA_FV_JSON_PATH);
    return 0;
}

void app_ipcam_FvMonitor_Stop(void) {
    if (!s_fv_run) {
        return;
    }
    s_fv_run = 0;
    /* detached thread: wait briefly for it to drain and clean up */
    for (int i = 0; i < 50 && s_fv_alive; i++) {
        usleep(20 * 1000);
    }
    unlink(CAMERA_FV_JSON_PATH);
    unlink(CAMERA_FV_TMP_PATH);
}
