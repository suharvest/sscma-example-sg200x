#ifndef _DEBUG_STREAM_H_
#define _DEBUG_STREAM_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * debug_stream: lazy H.264-over-WebSocket debug video + inference result
 * JSON fan-out, running inside the application process.
 *
 * - Own mg_mgr + single poll thread. Producer threads (VENC callback, the
 *   inference loop) never touch mongoose directly: they enqueue into a
 *   bounded queue and kick the poll thread via mg_wakeup().
 * - Video frame format on the wire (binary WS message):
 *       [Annex-B H.264 access unit bytes][uint64 unix-epoch milliseconds, LE]
 *   i.e. an 8-byte little-endian timestamp appended at the tail. The web
 *   client strips the tail (slice(0, -8)) before feeding JMuxer.
 * - Results path carries the inference result JSON as WS text messages.
 * - Lazy: with zero video clients debug_stream_video_handler() returns
 *   immediately (single atomic load, no locks, no copies).
 * - When a video client joins, an IDR is requested from the encoder and
 *   non-key frames are dropped until the SPS/PPS+IDR sequence is sent.
 */

typedef struct {
    int port;                  /* WS server port (default 8001) */
    const char* video_path;    /* WS path for H.264 video (default "/") */
    const char* results_path;  /* WS path for result JSON (default "/results") */
    int video_ch;              /* VENC channel for IDR requests (default VIDEO_CH2 = 2) */
    int max_video_clients;     /* default 2 */
    int max_results_clients;   /* default 2 */
} debug_stream_config_t;

/* Fill cfg with the defaults documented above. */
void debug_stream_config_init(debug_stream_config_t* cfg);

/* Start the WS server (spawns the poll thread). Returns 0 on success. */
int debug_stream_create(const debug_stream_config_t* cfg);

/* Stop the server, join the poll thread and release all resources. */
void debug_stream_destroy(void);

/*
 * pfpDataConsumes-compatible VENC stream consumer. Register it as a second
 * consumer on the RTSP channel (RTSP already owns index 0):
 *     registerVideoFrameHandler(VIDEO_CH2, 1, debug_stream_video_handler, NULL);
 * pData must be a VENC_STREAM_S*.
 */
int debug_stream_video_handler(void* pData, void* pCtx, void* pUserData);

/* Push one result JSON document to all /results clients (lazy, non-blocking). */
int debug_stream_publish_result(const char* json, size_t len);

/* Number of connected video / results WS clients. */
int debug_stream_video_client_count(void);
int debug_stream_results_client_count(void);

#ifdef __cplusplus
}
#endif

#endif /* _DEBUG_STREAM_H_ */
