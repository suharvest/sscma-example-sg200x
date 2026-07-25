#ifndef _RTSP_SERVER_H_
#define _RTSP_SERVER_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * rtsp_server: H.264/H.265 RTSP publishing for the gallery applications.
 *
 * Replaces the eight byte-identical copies of rtsp_demo.c that used to live in
 * solutions/<app>/main/. Those copies hard-coded port 8554, session name
 * "live%d" and bitrate 30720, and offered no authentication -- none of which
 * an ONVIF GetProfiles/GetStreamUri response can be built from. This component
 * keeps the same behaviour by default but makes all of it configurable and,
 * crucially, queryable.
 *
 * The streaming library (currently cvi_rtsp, which statically bundles
 * live555 2020.07.21) is an implementation detail and does not appear in this
 * header. That matters for two reasons: the shipped live555 carries
 * network-reachable CVEs and is due for replacement, and ONVIF Profile T needs
 * a third media track that cvi_rtsp's two-track struct cannot express.
 */

#define RTSP_SERVER_MAX_SESSIONS 6

typedef struct {
    /* Listen port. Default 8554.
     * NOTE: several apps still log "rtsp://<ip>:554/live" -- that string was
     * always wrong; the server has only ever listened on 8554. */
    int port;

    /* Session name prefix; session i is "<prefix><i>". Default "live". */
    const char* session_prefix;

    /* Bitrate hint in kbps. Only sizes live555's RTP send buffer, it does not
     * configure the encoder. Default 30720. */
    unsigned int bitrate;

    /* RTSP Basic/Digest credentials. Both NULL or empty -> no authentication,
     * which is the historical behaviour and remains the default. */
    const char* username;
    const char* password;

    /* Bitmask of VENC channels to publish: bit i -> channel i, in order.
     * Same encoding initRtsp() has always taken. */
    uint8_t ch_mask;
} rtsp_server_config_t;

/* Fill cfg with the defaults documented above. */
void rtsp_server_config_init(rtsp_server_config_t* cfg);

/* Create the server and one session per selected channel. Returns 0 on
 * success. Safe to call when already started (logs and returns 0). */
int rtsp_server_start(const rtsp_server_config_t* cfg);

/* Destroy every session and the server. Safe when not started. */
void rtsp_server_stop(void);

/*
 * pfpDataConsumes-compatible VENC stream consumer. Register as consumer
 * index 0 on each published channel:
 *     registerVideoFrameHandler(VIDEO_CH0, 0, rtsp_server_video_handler, NULL);
 * pData must be a VENC_STREAM_S*, pArgs an APP_DATA_CTX_S*.
 */
int rtsp_server_video_handler(void* pData, void* pArgs, void* pUserData);

/*
 * Self-description. ONVIF's GetProfiles/GetStreamUri have to answer with the
 * real port, session name and whether credentials are required, so the server
 * has to be able to describe itself rather than having those values duplicated
 * at every call site (which is how the ":554" bug survived).
 */
int rtsp_server_port(void);
int rtsp_server_session_count(void);
const char* rtsp_server_session_name(int idx); /* "live0"; NULL if out of range */
bool rtsp_server_auth_enabled(void);

/*
 * Build "rtsp://[user@]host:port/<session>" into buf. Returns the number of
 * bytes written (excluding NUL), or -1 on error. The password is never
 * included -- this is for logs, manifests and ONVIF responses.
 */
int rtsp_server_url(char* buf, size_t buflen, const char* host, int idx);

/*
 * Back-compat shims with the exact semantics of the old rtsp_demo.c, so a
 * solution migrates by deleting its rtsp_demo.{c,h}, switching the include and
 * adding rtsp_server to PRIVATE_REQUIREDS. New code should prefer
 * rtsp_server_start()/rtsp_server_stop()/rtsp_server_video_handler().
 */
int initRtsp(uint8_t chEnableFlag);
int deinitRtsp(void);
int fpStreamingSendToRtsp(void* pData, void* pArgs, void* pUserData);

#ifdef __cplusplus
}
#endif

#endif /* _RTSP_SERVER_H_ */
