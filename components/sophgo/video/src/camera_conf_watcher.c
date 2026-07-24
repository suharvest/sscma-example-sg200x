/* Background camera.conf watcher: hot-applies orientation changes to the
 * running VI channel. See app_ipcam_conf_watcher.h for the contract. */

#include "app_ipcam_conf_watcher.h"

#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "app_ipcam_camera_conf.h"
#include "app_ipcam_paramparse.h" /* APP_PROF_LOG_PRINT */

#include <cvi_vi.h>

#define CW_POLL_INTERVAL_US (1000 * 1000) /* 1 s */
#define CW_RETRY_BACKOFF_MS (30 * 1000)   /* 30 s after a CVI failure */

static volatile int s_cw_run   = 0;
static volatile int s_cw_alive = 0;
static int s_cw_pipe           = 0;
static int s_cw_chn            = 0;
static int s_cw_mirror_eff     = 0; /* effective conf values currently on VI */
static int s_cw_flip_eff       = 0;

static uint64_t cw_monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)(ts.tv_nsec / 1000000);
}

/* File identity stamp: mtime(sec,nsec) + size. Missing file -> all zero,
 * which is a valid state (defaults) distinct from any existing file. */
typedef struct {
    long long sec;
    long nsec;
    long long size;
    int exists;
} cw_stamp_t;

static void cw_stat(cw_stamp_t* st) {
    struct stat sb;
    if (stat(CAMERA_CONF_PATH, &sb) == 0) {
        st->sec    = (long long)sb.st_mtim.tv_sec;
        st->nsec   = (long)sb.st_mtim.tv_nsec;
        st->size   = (long long)sb.st_size;
        st->exists = 1;
    } else {
        memset(st, 0, sizeof(*st));
    }
}

static int cw_stamp_eq(const cw_stamp_t* a, const cw_stamp_t* b) {
    return a->sec == b->sec && a->nsec == b->nsec && a->size == b->size && a->exists == b->exists;
}

/* Apply the delta between the wanted effective conf values and the values
 * already folded into VI. Toggles the live hardware state, so the sensor's
 * base orientation and app-level preferences are preserved (same XOR
 * composition as the startup path in vi.c). Returns 0 on success. */
static int cw_apply(int mirror_eff, int flip_eff) {
    if (mirror_eff == s_cw_mirror_eff && flip_eff == s_cw_flip_eff) {
        return 0; /* nothing to do */
    }

    CVI_BOOL f = 0, m = 0;
    CVI_S32 ret = CVI_VI_GetChnFlipMirror((VI_PIPE)s_cw_pipe, (VI_CHN)s_cw_chn, &f, &m);
    if (ret != CVI_SUCCESS) {
        APP_PROF_LOG_PRINT(LEVEL_WARN, "conf watcher: CVI_VI_GetChnFlipMirror(%d,%d) failed with %#x\n", s_cw_pipe, s_cw_chn, ret);
        return -1;
    }

    CVI_BOOL tf = (flip_eff != s_cw_flip_eff) ? (CVI_BOOL)!f : f;
    CVI_BOOL tm = (mirror_eff != s_cw_mirror_eff) ? (CVI_BOOL)!m : m;
    if (tf != f || tm != m) {
        ret = CVI_VI_SetChnFlipMirror((VI_PIPE)s_cw_pipe, (VI_CHN)s_cw_chn, tf, tm);
        if (ret != CVI_SUCCESS) {
            APP_PROF_LOG_PRINT(LEVEL_WARN, "conf watcher: CVI_VI_SetChnFlipMirror(%d,%d,%d,%d) failed with %#x\n", s_cw_pipe, s_cw_chn, tf, tm, ret);
            return -1;
        }
    }

    APP_PROF_LOG_PRINT(LEVEL_INFO, "conf watcher: applied effective mirror=%d flip=%d (VI flip=%d mirror=%d)\n", mirror_eff, flip_eff, tf, tm);
    s_cw_mirror_eff = mirror_eff;
    s_cw_flip_eff   = flip_eff;
    return 0;
}

static void* cw_thread(void* arg) {
    (void)arg;
    cw_stamp_t last;
    int pending         = 1; /* first tick always re-checks: covers a conf */
    int warned          = 0; /* change during the startup window */
    uint64_t next_try_ms = 0;

    prctl(PR_SET_NAME, "CONF_WATCHER", 0, 0, 0);
    s_cw_alive = 1;
    cw_stat(&last);

    while (s_cw_run) {
        cw_stamp_t cur;
        cw_stat(&cur);
        if (!cw_stamp_eq(&cur, &last)) {
            last    = cur;
            pending = 1;
        }

        if (pending && cw_monotonic_ms() >= next_try_ms) {
            camera_conf_t conf;
            app_ipcam_CameraConf_Load(CAMERA_CONF_PATH, &conf); /* missing -> zeros */
            int rot180     = (conf.rotation == 180);
            int mirror_eff = ((conf.mirror != 0) != rot180); /* CAM_MIRROR XOR rot180 */
            int flip_eff   = ((conf.flip != 0) != rot180);   /* CAM_FLIP XOR rot180 */

            if (cw_apply(mirror_eff, flip_eff) == 0) {
                pending = 0;
                warned  = 0;
            } else {
                if (!warned) {
                    APP_PROF_LOG_PRINT(LEVEL_WARN, "conf watcher: apply failed, retry every %ds\n", CW_RETRY_BACKOFF_MS / 1000);
                    warned = 1;
                }
                next_try_ms = cw_monotonic_ms() + CW_RETRY_BACKOFF_MS;
            }
        }

        usleep(CW_POLL_INTERVAL_US);
    }

    s_cw_alive = 0;
    return NULL;
}

int app_ipcam_ConfWatcher_Start(int vi_pipe, int vi_chn, int applied_mirror_eff, int applied_flip_eff) {
    if (s_cw_run) {
        return 0; /* already running */
    }
    s_cw_pipe       = vi_pipe;
    s_cw_chn        = vi_chn;
    s_cw_mirror_eff = applied_mirror_eff ? 1 : 0;
    s_cw_flip_eff   = applied_flip_eff ? 1 : 0;
    s_cw_run        = 1;

    pthread_t tid;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    int ret = pthread_create(&tid, &attr, cw_thread, NULL);
    pthread_attr_destroy(&attr);
    if (ret != 0) {
        s_cw_run = 0;
        APP_PROF_LOG_PRINT(LEVEL_WARN, "create conf watcher thread failed (%d)\n", ret);
        return -1;
    }
    APP_PROF_LOG_PRINT(LEVEL_INFO, "conf watcher started on pipe %d chn %d (baseline mirror=%d flip=%d)\n", vi_pipe, vi_chn, s_cw_mirror_eff, s_cw_flip_eff);
    return 0;
}

void app_ipcam_ConfWatcher_Stop(void) {
    if (!s_cw_run) {
        return;
    }
    s_cw_run = 0;
    /* detached thread: wait briefly for it to drain (poll tick is 1 s) */
    for (int i = 0; i < 60 && s_cw_alive; i++) {
        usleep(20 * 1000);
    }
}
