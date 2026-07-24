/* Device-level camera.conf parser (pure C, libc only).
 * See app_ipcam_camera_conf.h for the file format. */

#include "app_ipcam_camera_conf.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Trim leading/trailing whitespace in place, return start pointer. */
static char* cc_trim(char* s) {
    while (*s && isspace((unsigned char)*s)) {
        s++;
    }
    char* end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) {
        end--;
    }
    *end = '\0';
    return s;
}

/* Strict "0"/"1"/... integer parse; returns -1 on non-numeric garbage. */
static int cc_parse_int(const char* s, long* out) {
    char* endp = NULL;
    if (*s == '\0') {
        return -1;
    }
    long v = strtol(s, &endp, 10);
    if (endp == NULL || *endp != '\0') {
        return -1;
    }
    *out = v;
    return 0;
}

int app_ipcam_CameraConf_Load(const char* path, camera_conf_t* conf) {
    if (conf == NULL) {
        return -1;
    }
    conf->mirror   = 0;
    conf->flip     = 0;
    conf->rotation = 0;

    if (path == NULL) {
        return -1;
    }

    FILE* fp = fopen(path, "r");
    if (fp == NULL) {
        return -1; /* missing file -> all defaults, not an error for callers */
    }

    char line[128];
    while (fgets(line, sizeof(line), fp) != NULL) {
        char* p = cc_trim(line);
        if (*p == '\0' || *p == '#') {
            continue; /* blank line or comment */
        }
        char* eq = strchr(p, '=');
        if (eq == NULL) {
            continue; /* not KEY=VALUE */
        }
        *eq       = '\0';
        char* key = cc_trim(p);
        char* val = cc_trim(eq + 1);

        long v = 0;
        if (cc_parse_int(val, &v) != 0) {
            continue; /* bad value -> keep default */
        }

        /* whitelist */
        if (strcmp(key, "CAM_MIRROR") == 0) {
            if (v == 0 || v == 1) {
                conf->mirror = (int)v;
            }
        } else if (strcmp(key, "CAM_FLIP") == 0) {
            if (v == 0 || v == 1) {
                conf->flip = (int)v;
            }
        } else if (strcmp(key, "CAM_ROTATION") == 0) {
            if (v == 0 || v == 180) {
                conf->rotation = (int)v;
            }
        }
        /* unknown keys ignored */
    }

    fclose(fp);
    return 0;
}
