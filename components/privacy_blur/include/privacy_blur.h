#ifndef _PRIVACY_BLUR_H_
#define _PRIVACY_BLUR_H_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sscma.h>

#include "norm_box.h"

/*
 * Privacy masking for the encoded video stream, shared by every application
 * that wants to conceal what it detects.
 *
 * This started life inside detection-blur and moved out unchanged in substance:
 * the three backends, the RGB332 packing, the LUT upload and the Kalman tracker
 * below are all code that was validated on real hardware, and they were carried
 * over rather than rewritten.
 *
 * Deliberately no CVI type appears in this header. An application asks for
 * boxes to be hidden; whether that happens through a MOSAIC region, a COVEREX
 * rectangle or a composited ARGB bitmap is a detail of the CV181x region engine
 * that no application has a reason to know, and keeping it out means adding a
 * fourth backend does not touch a single application.
 */

namespace privacy_blur {

/*
 * Internal working form: normalised against the video stream, centre-based.
 *
 * Applications do not construct these -- the public entry points take
 * geometry::StreamBox and geometry::InferBox, which say so in their type. This
 * struct stays because the tracker and the LUT builder below were validated on
 * hardware against exactly these field names, and retyping them buys nothing
 * that the API boundary has not already bought.
 *
 * Deliberately carries no class id: deciding *what* is worth hiding is the
 * application's judgement, and an application that only wants to mask people
 * simply does not pass the cars.
 */
struct BlurBox {
    float x = 0.0f;  /* centre x, [0,1] */
    float y = 0.0f;  /* centre y, [0,1] */
    float w = 0.0f;  /* width,    [0,1] */
    float h = 0.0f;  /* height,   [0,1] */
    float score = 0.0f;
};

/*
 * Pixelate `boxes` directly in an RGB888 buffer, in place.
 *
 * For the outputs the hardware mask cannot reach. The RGN mask lives in the
 * VPSS->VENC path, so it covers RTSP and the console's debug video and nothing
 * else; a JPEG snapshot encoded from the inference frame goes out completely
 * unmasked. That is not a cosmetic gap -- the snapshot URL is what ONVIF
 * advertises as GetSnapshotUri, so a client that respects the mask on the video
 * stream can pull an unmasked still of the same scene from the same device.
 *
 * Takes InferBox because the buffer being pixelated is the inference frame --
 * that is what a snapshot is encoded from, and boxes must be normalised against
 * the buffer in hand, not against the stream. Passing a StreamBox here does not
 * compile, which is the point: assuming a relationship between two differently
 * shaped frames is exactly the bug that put the hardware mask 16/9 too tall.
 *
 * Cost is paid only where it is asked for: the loop touches the pixels inside
 * the boxes and nothing else, and callers should only invoke it when a snapshot
 * is actually due (debug_stream_snapshot_armed()).
 */
void pixelateRgb888(void* rgb888, int width, int height,
                    const std::vector<geometry::InferBox>& boxes, int block_px);

/*
 * Device-wide privacy settings, written by supervisor and read in-process by
 * whichever application is running. Same mechanism as /userdata/local/ha.conf
 * and /userdata/local/onvif.conf: one switch in the console, no per-application
 * configuration and no init-script plumbing to touch.
 *
 * File: /userdata/local/blur.conf, shell-sourceable KEY='value' lines.
 *
 *   BLUR_ENABLED=0|1
 *   BLUR_BACKEND=mosaic|coverex|pixelate
 *   BLUR_BLOCK_PX=8|16
 *   BLUR_MAX_REGIONS=8
 *   BLUR_ALPHA=0..255
 */
struct PrivacyBlurConfig {
    bool present = false;  /* file existed and parsed */

    /* Off unless someone asked for it. A privacy mask changes what the stream
     * shows, so a device that was never configured must not start hiding parts
     * of its own video -- that is the operator's decision to make. */
    bool enabled = false;

    /*
     * "pixelate" by default because it is the only backend that conceals
     * without also destroying the picture: it averages the pixels it hides, so
     * the mask keeps the subject's luminance and hue while losing every feature
     * smaller than a block. "mosaic" renders as television static on a stock
     * kernel (the CV181x driver fills its grid from get_random_u32()) and
     * "coverex" paints a flat rectangle, which reads as a censorship bar.
     */
    std::string backend = "pixelate";

    /*
     * Edge length of one mosaic block, in stream pixels. The hardware MOSAIC
     * unit only knows 8 and 16, so anything else is clamped to 16 rather than
     * rejected -- a hand-edited file must not be able to leave a device with no
     * mask at all.
     */
    int block_px = 16;

    /*
     * Upper bound on how many mosaic blocks may span a concealed subject.
     * 0 keeps the fixed block_px and the behaviour that existed before.
     *
     * A block size fixed in pixels protects unevenly, and it fails in the worst
     * direction: a face 200 px across is covered by a dozen blocks and is
     * unreadable, while the same face at arm's length spans 800 px, is covered
     * by fifty, and keeps enough structure to show the eyes, the mouth and the
     * shape of the jaw. The subject who is nearest the camera -- the one most
     * identifiable to begin with -- is the one the mask protects least.
     *
     * Capping the block count instead makes concealment independent of how
     * close the subject stands: the blocks grow with the box, so a face is
     * always reduced to about the same amount of information.
     *
     * The hardware has no say in this. Its mosaic unit only offers 8x8 and
     * 16x16 cells, so a coarser look cannot be asked of it directly; what makes
     * this work is that we author the colour table ourselves, and giving a
     * kxk group of cells one shared colour reads exactly as a 16k-pixel block.
     * The cost is one extra average over values already computed.
     */
    int blocks_per_target = 12;

    /* How many regions may be concealed at once. See PrivacyBlur::init() for
     * what happens when a scene contains more than this. */
    int max_regions = 8;

    /*
     * Opacity of the hardware mask, 0 (invisible) to 255 (nothing shows
     * through).
     *
     * Fully opaque by default, and that is not a compromise waiting to be
     * tuned: whatever fraction of the original picture the mask lets through is
     * exactly the fraction of identifying detail it hands back, so a mask soft
     * enough to look nice is a mask the subject can be recognised through, and
     * at that point it is no longer a privacy feature. Turning it down is a
     * reasonable thing for an operator to want -- to demonstrate that the mask
     * is tracking the right subject, or because their own rules ask for a
     * visible-but-obscured image -- but it has to be their explicit decision
     * against their own requirements.
     *
     * Only the hardware mosaic path honours this, because it is the kernel's
     * privacy-mask unit that does the blending. The software compositing path
     * paints opaque pixels and has nothing to blend with.
     */
    int alpha = 255;
};

/*
 * Parse the config file. A missing file is not an error: `out` comes back with
 * present=false, enabled=false and the documented defaults, which is exactly
 * the "privacy blur off" state. Returns false only on a genuine parse failure.
 */
bool loadPrivacyBlurConfig(const std::string& path, PrivacyBlurConfig& out,
                           std::string* error = nullptr);

/* Default location, matching what supervisor writes. */
extern const char* const PRIVACY_BLUR_CONFIG_PATH;

// ---------------------------------------------------------------------------

/* 1D Kalman filter with a constant-velocity model.
 * State: [position, velocity], observation: [position]. */
struct KalmanFilter1D {
    float x;    // position estimate
    float v;    // velocity estimate
    float p00;  // var(position)
    float p01;  // cov(position, velocity)
    float p11;  // var(velocity)

    void init(float x0, float pos_var = 0.01f, float vel_var = 1.0f);
    void predict(float dt, float q);
    void update(float z, float r);
};

/* Tracked bounding box with 4 independent Kalman filters (x, y, w, h). */
struct TrackedRegion {
    KalmanFilter1D kf[4];  // center_x, center_y, width, height
    float score;
    int miss_count;

    void init(const BlurBox& box);
    void predict(float dt, float q);
    void update(const BlurBox& box, float r);
    BlurBox getBox() const;
};

class PrivacyBlur {
public:
    PrivacyBlur();
    ~PrivacyBlur();

    /*
     * Bring the masking hardware up. Must be called AFTER the video pipeline is
     * running, because RGN regions attach to a live VPSS channel.
     *
     * Returns false when no region could be created, which is the honest signal
     * that this application will stream unmasked video; callers should treat it
     * as a warning rather than a fatal error, since detection itself still
     * works.
     *
     * Passing a config with enabled=false still initialises everything and
     * simply starts in the hidden state, so setEnabled(true) later is instant.
     */
    bool init(const PrivacyBlurConfig& cfg, int stream_w, int stream_h,
              int vpss_grp = 0, int vpss_chn = 2);

    void deinit();

    /*
     * Runtime switch. Turning it off hides the masks but keeps the regions and
     * the tracker alive, so toggling back on is immediate and never disturbs
     * the video pipeline -- restarting the application to compare masked with
     * unmasked output would change the scene between the two captures, which is
     * exactly what has to stay constant to judge a mask.
     */
    void setEnabled(bool on);
    bool enabled() const { return enabled_.load(); }

    /* Block size in use, so a caller masking an output the hardware cannot
     * reach (the JPEG snapshot) can match the look of the video mask instead of
     * picking a second, differing constant. */
    int blockPx() const { return block_px_; }

    /*
     * Feed detections.
     *
     * `frame` is the RGB888 inference frame and may be null. It is needed by
     * the pixelating backends, which average the pixels they are about to hide;
     * hardware MOSAIC cannot do that and COVEREX only paints a flat colour.
     * Passing the frame is what makes the mask look like the scene behind it
     * instead of a censorship bar.
     *
     * Must be called BEFORE the frame is returned to the camera.
     *
     * Takes StreamBox: the mask is composited into the VPSS->VENC path, so the
     * boxes must be normalised against the stream. Detections normalised
     * against the inference channel go through geometry::toStream() first --
     * and because that is the only way to obtain a StreamBox from an InferBox,
     * skipping it is a compile error rather than a mask that covers three
     * quarters of the face.
     */
    void onDetection(const std::vector<geometry::StreamBox>& boxes,
                     const ma_img_t* frame = nullptr);

private:
    void initRegions();
    void deinitRegions();
    void applyRegions(const std::vector<BlurBox>& boxes, const ma_img_t* frame);
    /* Keep only as many boxes as the configured capacity, largest first. */
    void truncateToCapacity(std::vector<BlurBox>& boxes);
    /* Fill and upload the hardware colour table for the grid the driver
     * reports. Returns false when the kernel has no colour-mosaic support, in
     * which case the mask stays whatever the driver draws on its own. */
    /* `boxes` is needed as well as the frame: the block size is chosen per
     * subject, so filling the table has to know which cells belong to which
     * box. Cells outside every box keep their own per-cell colour. */
    bool updateHwLut(const ma_img_t* frame, const std::vector<BlurBox>& boxes,
                     int target_w, int target_h,
                     int offset_x, int offset_y);
    bool renderPixelated(const std::vector<BlurBox>& boxes, int active_count,
                         const ma_img_t* frame, int target_w, int target_h,
                         int offset_x, int offset_y);
    void associateAndUpdate(const std::vector<BlurBox>& boxes);
    float computeIoU(const BlurBox& a, const BlurBox& b);
    void predictThreadEntry();

private:
    static constexpr int kRgnHandleBase = 100;
    static constexpr int kMaxRegionsLimit = 8;    // RGN_MOSAIC_MAX_NUM per channel on CV181x
    static constexpr int kMaxCoverexRegions = 4;  // COVEREX_RGN max per channel

    PrivacyBlurConfig cfg_;

    /* How many detections may be concealed at once. Not always the same as the
     * number of RGN regions: the software pixelating backend composites every
     * mask into a single region, so it has one handle but a capacity of
     * several. */
    int capacity_;
    /* How many RGN regions are actually created. */
    int region_count_;

    int vpss_grp_;
    int vpss_chn_;
    /* RGN_HANDLE is a CVI_S32; spelled as int32_t so this header stays free of
     * the CVI SDK. */
    std::vector<int32_t> handles_;
    bool use_coverex_ = false;
    bool use_overlayex_ = false;
    /* Hardware colour mosaic: MOSAIC regions as usual, but the per-cell colour
     * table is supplied through the patched driver's RGN_SDK_SET_MOSAIC_LUT so
     * the scaler composites it. Costs a few hundred bytes a frame instead of
     * the 3.6 MB the software OVERLAY path uploads. */
    bool use_hw_pixelate_ = false;
    int rgn_fd_ = -1;
    int lut_test_ = 0;
    uint8_t frame_avg_ = 0x92;
    std::vector<uint8_t> lut_;
    /* Per-cell RGB averages, kept between the two passes of updateHwLut(). */
    std::vector<uint8_t> cell_rgb_;
    /* Scratch for the ARGB tile, reused across frames and regions so the hot
     * path does no allocation. */
    std::vector<uint32_t> tile_;
    /* Edge length in stream pixels of one mosaic block. Bigger hides more. */
    int block_px_ = 16;
    /* Whether a tile has ever been uploaded to each region. An OVERLAY region
     * with no bitmap composites as nothing, so showing one before its first
     * render would silently disable the mask. */
    std::vector<bool> tile_uploaded_;
    bool regions_inited_;

    /* Frames handed to onDetection() since init, and the window during which a
     * near-full-frame box is treated as ISP start-up noise rather than as a
     * subject standing close to the camera. Counted rather than timed because
     * what matters is how many frames the ISP has produced, not how long the
     * process has been alive. */
    static constexpr unsigned kIspSettleFrames = 30;

    /* How far a subject may have moved between detections and still be
     * recognised as the same one, in multiples of its own width. Just over one
     * body-width: far enough to follow someone walking briskly at 5 detections
     * a second, short enough that two different people standing apart are never
     * confused for one. */
    static constexpr float kMaxAssocDist = 1.5f;
    unsigned frames_seen_ = 0;
    bool large_box_warned_ = false;

    /* Runtime switch, read on the detection thread and on the prediction
     * thread, so atomic rather than mutex-guarded: it gates work, it does not
     * order it. */
    std::atomic<bool> enabled_;

    /* How many boxes have been dropped for want of capacity since start-up,
     * and whether the first drop has been reported. Counted rather than logged
     * per frame because a crowded scene would otherwise fill the log with the
     * same line fifteen times a second. */
    uint64_t dropped_total_ = 0;
    bool drop_reported_ = false;

    // Stream resolution
    int stream_width_;
    int stream_height_;

    // Kalman prediction tracking
    std::vector<TrackedRegion> trackers_;
    std::mutex tracker_mutex_;
    std::thread predict_thread_;
    std::atomic<bool> predicting_;
    float process_noise_;
    float measurement_noise_;
    int max_miss_;
    int predict_interval_ms_;
    float iou_threshold_;

    bool initialized_;
};

}  // namespace privacy_blur

#endif  // _PRIVACY_BLUR_H_
