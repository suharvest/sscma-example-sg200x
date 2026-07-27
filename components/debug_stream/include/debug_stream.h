#ifndef _DEBUG_STREAM_H_
#define _DEBUG_STREAM_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * debug_stream: lazy H.264-over-WebSocket debug video + inference result
 * JSON fan-out, running inside the application process.
 *
 * - Own listener + single event thread (see src/ws_transport.h; the HTTP/WS
 *   library is confined to one backend file). Producer threads (VENC callback,
 *   the inference loop) never touch the library: they enqueue into a bounded
 *   queue and kick the event thread via ws_transport_wake().
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
    const char* snapshot_path; /* HTTP path for JPEG snapshot (default "/snapshot.jpg") */
    int video_ch;              /* VENC channel for IDR requests (default VIDEO_CH2 = 2) */
    int max_video_clients;     /* default 2 */
    int max_results_clients;   /* default 2 */
} debug_stream_config_t;

/* Fill cfg with the defaults documented above. */
void debug_stream_config_init(debug_stream_config_t* cfg);

/* Start the WS server (spawns the poll thread). Returns 0 on success. */
int debug_stream_create(const debug_stream_config_t* cfg);

/*
 * Convenience bring-up used by the gallery applications: default config +
 * port/video_ch override + create. On success it logs the ws:// URLs and
 * registers debug_stream_video_handler as VENC consumer index 1 on video_ch
 * (RTSP owns index 0), then returns 0. On failure it logs a warning and
 * returns non-zero so the caller can degrade (run without the debug stream).
 */
int debug_stream_start_or_disable(int port, int video_ch);

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

/*
 * JPEG snapshot of the live scene, served as GET <snapshot_path>
 * (default "/snapshot.jpg"). Needed by ONVIF GetSnapshotUri, and useful on its
 * own for console thumbnails and post-deploy verification.
 *
 * Lazy in the same spirit as the video path, because encoding is not free on
 * this SoC: nothing is copied or encoded until a client actually asks. A GET
 * "arms" the snapshot for DEBUG_STREAM_SNAPSHOT_ARM_MS; while armed, the
 * producer's offer calls encode at most one frame per
 * DEBUG_STREAM_SNAPSHOT_MIN_INTERVAL_MS. Once nobody has asked for a while it
 * goes back to costing a single atomic load per frame.
 *
 * Consequence worth knowing: the first GET after an idle period has nothing to
 * serve yet and gets 503 with Retry-After: 1. Steady-state polling (which is
 * how both ONVIF clients and the console use it) always sees 200.
 */
#define DEBUG_STREAM_SNAPSHOT_ARM_MS 10000
#define DEBUG_STREAM_SNAPSHOT_MIN_INTERVAL_MS 500

/*
 * Offer the current frame for snapshotting. Call once per processed frame from
 * the inference thread, with the same RGB888 buffer that was just inferred on.
 * Returns immediately (one atomic load) unless a snapshot is due.
 *
 * Encoding happens inline on the calling thread -- deliberately, so the
 * transport's event thread never runs cv::imencode. Do not call this from the
 * VENC callback: that runs at real-time priority and starving it is what broke
 * RTSP once already.
 */
int debug_stream_offer_snapshot(const void* rgb888, int width, int height);

/* Whether a snapshot client has asked recently, i.e. whether the next
 * debug_stream_offer_snapshot() may do real work. */
bool debug_stream_snapshot_armed(void);

#ifdef __cplusplus
}

#include <string>
#include <vector>

/*
 * One detection box for the /results envelope. Coordinates are center-based
 * pixels in the inference resolution; label is rendered verbatim by the
 * console overlay (BoxOverlay reads box[5] as the on-screen text).
 */
struct debug_stream_box_t {
    float x;
    float y;
    float w;
    float h;
    float score;
    std::string label;
};

/*
 * Build the sscma-node compatible result JSON for the debug /results
 * channel:
 *   {"timestamp":..,"frame_id":..,"inference_time_ms":..,
 *    "resolution":[w,h],"boxes":[[x,y,w,h,score,"label"],..],
 *    "labels":["..",..](, <extra_json>)}
 * labels is the parallel array (labels[i] <-> boxes[i]) for programmatic
 * consumers; pass nullptr to mirror each box's label. extra_json, when
 * non-empty, is appended verbatim as additional top-level members (e.g.
 * "\"zone\":{...}").
 * NOTE: this is a separate document from the MQTT payload; the MQTT format
 * is an external contract and must not change.
 */
std::string debug_stream_build_results(uint64_t timestamp_ms, uint32_t frame_id,
                                       float inference_time_ms, int res_w, int res_h,
                                       const std::vector<debug_stream_box_t>& boxes,
                                       const std::vector<std::string>* labels = nullptr,
                                       const std::string& extra_json = std::string());

/*
 * Remap overlay boxes from the letterboxed inference frame into the display
 * (stream) frame, then report with res = display dims.
 *
 * The camera VPSS fits the sensor content into each channel preserving aspect
 * (ASPECT_RATIO_AUTO), padding with bars. When the inference channel aspect
 * differs from the debug-video (stream) channel aspect, a box built in
 * inference-frame pixels carries a different letterbox offset than the video,
 * so drawn over the video it is misplaced (e.g. a square 640x640 inference vs a
 * 16:9 stream). This maps each box (given in inference-frame pixels) into the
 * display frame the debug video actually shows: it assumes the display channel
 * shows the un-letterboxed sensor content (display aspect == sensor aspect,
 * true for a full-FOV 16:9 stream). When the two aspects already match it is a
 * pure scale (a no-op when the dims are equal). Call this, then pass the boxes
 * to debug_stream_build_results with res_w/res_h = display_w/display_h.
 */
void debug_stream_letterbox_to_display(std::vector<debug_stream_box_t>& boxes,
                                       int inference_w, int inference_h,
                                       int display_w, int display_h);

/*
 * The inverse: display-frame pixels back into the letterboxed inference frame.
 *
 * Needed by anything that has to draw on the inference frame using coordinates
 * expressed against the video stream -- masking the JPEG snapshot, for one,
 * since the snapshot is encoded from the inference frame while the privacy
 * mask's boxes are normalised against the stream.
 *
 * Kept next to its forward counterpart, and sharing its arithmetic, so the two
 * cannot drift apart. Two hand-written copies of this mapping is how the mask
 * and the overlay ended up disagreeing about where a face was.
 */
void debug_stream_display_to_letterbox(std::vector<debug_stream_box_t>& boxes,
                                       int inference_w, int inference_h,
                                       int display_w, int display_h);
#endif

#endif /* _DEBUG_STREAM_H_ */
