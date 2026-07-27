#include "privacy_blur.h"
#include "mosaic_lut.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include <sys/stat.h>

#include <sscma.h>
#include <cvi_region.h>

#define TAG "PrivacyBlur"

namespace privacy_blur {

namespace {

/* Where the patched cv181x_vpss driver exposes the privacy mask's opacity. */
const char* const MASK_ALPHA_SYSFS = "/sys/module/cv181x_vpss/parameters/mask_alpha";

/*
 * Push the configured opacity into the kernel.
 *
 * Reports whether the value actually landed, so the caller can say so once and
 * then stop talking about it. A kernel without this parameter is not a fault
 * condition: it is simply an older driver whose mask is always fully opaque,
 * which is the safe behaviour anyway, and an application that logged an error
 * every time it started on such a device would be crying wolf.
 */
bool writeMaskAlpha(int alpha)
{
    if (alpha < 0) alpha = 0;
    if (alpha > 255) alpha = 255;

    FILE* f = fopen(MASK_ALPHA_SYSFS, "w");
    if (f == nullptr) return false;
    const int written = fprintf(f, "%d\n", alpha);
    const bool ok = (fclose(f) == 0) && (written > 0);
    return ok;
}

}  // namespace

// ============ KalmanFilter1D ============
// Constant-velocity model: F = [[1,dt],[0,1]], H = [1,0]
// Process noise: Q = q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]

void KalmanFilter1D::init(float x0, float pos_var, float vel_var) {
    x   = x0;
    v   = 0.0f;
    p00 = pos_var;
    p01 = 0.0f;
    p11 = vel_var;
}

void KalmanFilter1D::predict(float dt, float q) {
    x += v * dt;

    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt2 * dt2;

    p00 = p00 + 2.0f * p01 * dt + p11 * dt2 + q * dt4 * 0.25f;
    p01 = p01 + p11 * dt + q * dt3 * 0.5f;
    p11 = p11 + q * dt2;
}

void KalmanFilter1D::update(float z, float r) {
    float y  = z - x;
    float s  = p00 + r;
    float k0 = p00 / s;
    float k1 = p01 / s;

    x += k0 * y;
    v += k1 * y;

    float new_p00 = p00 - k0 * p00;
    float new_p01 = p01 - k0 * p01;
    float new_p11 = p11 - k1 * p01;

    p00 = new_p00;
    p01 = new_p01;
    p11 = new_p11;
}

// ============ TrackedRegion ============

void TrackedRegion::init(const BlurBox& box) {
    kf[0].init(box.x);
    kf[1].init(box.y);
    kf[2].init(box.w, 0.01f, 0.1f);
    kf[3].init(box.h, 0.01f, 0.1f);
    score      = box.score;
    miss_count = 0;
}

void TrackedRegion::predict(float dt, float q) {
    for (int i = 0; i < 4; i++) {
        kf[i].predict(dt, q);
    }
}

void TrackedRegion::update(const BlurBox& box, float r) {
    kf[0].update(box.x, r);
    kf[1].update(box.y, r);
    kf[2].update(box.w, r * 2.0f);
    kf[3].update(box.h, r * 2.0f);
    score      = box.score;
    miss_count = 0;
}

BlurBox TrackedRegion::getBox() const {
    BlurBox b;
    b.x     = kf[0].x;
    b.y     = kf[1].x;
    b.w     = std::max(0.01f, kf[2].x);
    b.h     = std::max(0.01f, kf[3].x);
    b.score = score;
    return b;
}

// ============ PrivacyBlur ============

PrivacyBlur::PrivacyBlur()
    : capacity_(kMaxRegionsLimit),
      region_count_(kMaxRegionsLimit),
      vpss_grp_(0),
      vpss_chn_(2),
      regions_inited_(false),
      enabled_(false),
      stream_width_(0),
      stream_height_(0),
      predicting_(false),
      process_noise_(5.0f),
      measurement_noise_(0.001f),
      max_miss_(15),
      predict_interval_ms_(33),
      iou_threshold_(0.2f),
      initialized_(false) {}

PrivacyBlur::~PrivacyBlur() {
    deinit();
}

void PrivacyBlur::setEnabled(bool on) {
    enabled_.store(on);
}

bool PrivacyBlur::init(const PrivacyBlurConfig& cfg, int stream_width, int stream_height,
                       int vpss_grp, int vpss_chn) {
    if (initialized_) return true;

    cfg_           = cfg;
    stream_width_  = stream_width;
    stream_height_ = stream_height;
    vpss_grp_      = vpss_grp;
    vpss_chn_      = vpss_chn;
    enabled_.store(cfg.enabled);

    if (stream_width_ <= 0 || stream_height_ <= 0) {
        MA_LOGE(TAG, "Invalid stream resolution: %dx%d", stream_width_, stream_height_);
        return false;
    }

    MA_LOGI(TAG, "Initializing privacy blur: stream %dx%d, vpss(%d,%d), backend=%s, "
                 "block=%dpx, max_regions=%d, enabled=%d",
            stream_width_, stream_height_, vpss_grp_, vpss_chn_,
            cfg_.backend.c_str(), cfg_.block_px, cfg_.max_regions, (int)cfg_.enabled);

    initRegions();

    if (!regions_inited_) {
        MA_LOGW(TAG, "No RGN regions available, privacy blur disabled");
        return false;
    }

    /*
     * Only the hardware path blends, so only the hardware path has an opacity
     * to set. Said once at start-up rather than per frame: the parameter is
     * global to the driver and nothing in the frame loop changes it.
     */
    if (use_hw_pixelate_) {
        if (writeMaskAlpha(cfg_.alpha)) {
            /*
             * Logged, not warned about. A partly transparent mask is a
             * deliberate trade between how natural the picture looks and how
             * much of the subject survives, the operator makes it with the
             * debug stream in front of them, and it is legitimate at values
             * this component has no way to second-guess.
             */
            MA_LOGI(TAG, "mask opacity %d/255", cfg_.alpha);
        } else if (cfg_.alpha != 255) {
            MA_LOGW(TAG, "this kernel has no %s, so the mask stays fully opaque and "
                         "BLUR_ALPHA=%d has no effect", MASK_ALPHA_SYSFS, cfg_.alpha);
        }
    }

    predicting_.store(true);
    predict_thread_ = std::thread(&PrivacyBlur::predictThreadEntry, this);

    initialized_ = true;
    MA_LOGI(TAG, "Privacy blur initialized with %d regions (capacity %d)",
            (int)handles_.size(), capacity_);
    return true;
}

void PrivacyBlur::deinit() {
    if (!initialized_) return;

    predicting_.store(false);
    if (predict_thread_.joinable()) {
        predict_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(tracker_mutex_);
        trackers_.clear();
    }

    deinitRegions();

    initialized_ = false;
    MA_LOGI(TAG, "Privacy blur deinitialized");
}

// ============ IoU computation ============

float PrivacyBlur::computeIoU(const BlurBox& a, const BlurBox& b) {
    float a_l = a.x - a.w * 0.5f, a_r = a.x + a.w * 0.5f;
    float a_t = a.y - a.h * 0.5f, a_b = a.y + a.h * 0.5f;
    float b_l = b.x - b.w * 0.5f, b_r = b.x + b.w * 0.5f;
    float b_t = b.y - b.h * 0.5f, b_b = b.y + b.h * 0.5f;

    float inter_w = std::max(0.0f, std::min(a_r, b_r) - std::max(a_l, b_l));
    float inter_h = std::max(0.0f, std::min(a_b, b_b) - std::max(a_t, b_t));
    float inter   = inter_w * inter_h;

    float uni = a.w * a.h + b.w * b.h - inter;
    return (uni > 1e-6f) ? inter / uni : 0.0f;
}

// ============ Data association + Kalman update ============

void PrivacyBlur::associateAndUpdate(const std::vector<BlurBox>& filtered) {
    std::vector<bool> det_matched(filtered.size(), false);

    /*
     * Greedy matching, on overlap first and on proximity when overlap fails.
     *
     * IoU alone cannot follow a subject that moves quickly. It is exactly zero
     * for boxes that do not touch, so it carries no information about whether
     * two non-overlapping boxes are the same face a moment apart or two
     * different faces -- and with detection arriving every 150-300 ms, a head
     * can easily travel more than its own width between frames. The old
     * detection then fails to match, a second tracker is created for what is
     * really the same person, and the first one goes on being drawn from its
     * stale position until it times out: the trailing mask that gave this away.
     *
     * So when nothing overlaps, fall back to distance between centres, measured
     * in units of the box's own size. That scales correctly with the subject:
     * a face near the camera is allowed to travel further in pixels than a
     * distant one, because both moved the same amount relative to themselves.
     * Sizes must also stay comparable, which stops a small face being matched
     * to a large one that happens to be nearby.
     */
    for (auto& tracker : trackers_) {
        BlurBox predicted = tracker.getBox();
        float best_iou = iou_threshold_;
        int best_idx   = -1;

        for (int d = 0; d < (int)filtered.size(); d++) {
            if (det_matched[d]) continue;
            float iou = computeIoU(predicted, filtered[d]);
            if (iou > best_iou) {
                best_iou = iou;
                best_idx = d;
            }
        }

        if (best_idx < 0) {
            float best_dist = kMaxAssocDist;
            for (int d = 0; d < (int)filtered.size(); d++) {
                if (det_matched[d]) continue;
                const BlurBox& det = filtered[d];

                const float scale = std::max(0.01f, (predicted.w + det.w) * 0.5f);
                const float dx = (predicted.x - det.x) / scale;
                const float dy = (predicted.y - det.y) / scale;
                const float dist = std::sqrt(dx * dx + dy * dy);
                if (dist >= best_dist) continue;

                /* Same subject, so roughly the same size. Without this a mask
                 * could jump from a face to a different, closer one. */
                const float sw = det.w > 0.0f ? predicted.w / det.w : 0.0f;
                const float sh = det.h > 0.0f ? predicted.h / det.h : 0.0f;
                if (sw < 0.5f || sw > 2.0f || sh < 0.5f || sh > 2.0f) continue;

                best_dist = dist;
                best_idx  = d;
            }
        }

        if (best_idx >= 0) {
            tracker.update(filtered[best_idx], measurement_noise_);
            det_matched[best_idx] = true;
        } else {
            tracker.miss_count++;
        }
    }

    // Remove dead trackers first to free up slots
    trackers_.erase(
        std::remove_if(trackers_.begin(), trackers_.end(),
            [this](const TrackedRegion& t) { return t.miss_count > max_miss_; }),
        trackers_.end());

    // Create new trackers for unmatched detections
    for (int d = 0; d < (int)filtered.size(); d++) {
        if (det_matched[d]) continue;

        if ((int)trackers_.size() < capacity_) {
            // Slot available: create new tracker
            TrackedRegion tr;
            tr.init(filtered[d]);
            trackers_.push_back(tr);
        } else {
            // All slots occupied: replace the tracker with highest miss_count
            int worst_idx = -1;
            int worst_miss = 0;
            for (int t = 0; t < (int)trackers_.size(); t++) {
                if (trackers_[t].miss_count > worst_miss) {
                    worst_miss = trackers_[t].miss_count;
                    worst_idx  = t;
                }
            }
            if (worst_idx >= 0 && worst_miss > 0) {
                trackers_[worst_idx].init(filtered[d]);
            }
        }
    }
}

// ============ Capacity truncation ============

/*
 * Drop the surplus when a scene contains more to hide than the hardware can
 * hide, keeping the largest boxes.
 *
 * Area is the right criterion because a bigger box means the subject is closer
 * to the lens, and a subject closer to the lens is the one a viewer can
 * actually recognise. A distant face twenty pixels across carries almost no
 * identifying detail whether it is masked or not, so when something has to go
 * unmasked it should be that one. Sorting by confidence instead -- which is
 * what the tracker does further down, for a different purpose -- would happily
 * leave the nearest face in the clear because the detector happened to be a
 * little less sure about it.
 *
 * The overflow itself is counted rather than logged per frame: a crowded scene
 * would otherwise repeat the same line at the detection rate. The first
 * occurrence is reported immediately so the condition is visible at all, and
 * the running total every thousand drops after that, so a long run still shows
 * how often it happened without the log ever becoming a flood.
 */
void PrivacyBlur::truncateToCapacity(std::vector<BlurBox>& boxes) {
    if ((int)boxes.size() <= capacity_) return;

    const size_t dropped = boxes.size() - (size_t)capacity_;

    std::partial_sort(boxes.begin(), boxes.begin() + capacity_, boxes.end(),
        [](const BlurBox& a, const BlurBox& b) {
            return (a.w * a.h) > (b.w * b.h);
        });
    boxes.resize((size_t)capacity_);

    const uint64_t before = dropped_total_;
    dropped_total_ += dropped;
    if (!drop_reported_) {
        drop_reported_ = true;
        MA_LOGW(TAG, "More regions to mask than capacity %d: the %zu smallest box(es) "
                     "are left unmasked. Further occurrences are counted, not logged.",
                capacity_, dropped);
    } else if ((before / 1000) != (dropped_total_ / 1000)) {
        MA_LOGW(TAG, "%llu boxes left unmasked so far for want of capacity (%d)",
                (unsigned long long)dropped_total_, capacity_);
    }
}

// ============ Prediction thread ============

void PrivacyBlur::predictThreadEntry() {
    MA_LOGI(TAG, "Prediction thread started (%d ms interval, q=%.3f, r=%.4f)",
            predict_interval_ms_, process_noise_, measurement_noise_);

    float dt = (float)predict_interval_ms_ / 1000.0f;

    /*
     * The settings an operator adjusts while watching the stream -- is masking
     * on, how opaque, how coarse -- are re-read here instead of only at
     * start-up.
     *
     * Restarting the application to apply them technically works and is what
     * used to happen, but it drops the video for several seconds, and the video
     * is the only way to judge whether the new value is right. Nobody can tune
     * an opacity through a viewfinder that goes black every time they move the
     * slider.
     *
     * Only the fields that can change without rebuilding anything are picked up
     * here. Backend, block size and region count decide how the RGN regions are
     * allocated, so those still need a restart and supervisor still performs
     * one for them.
     *
     * Polled on the tick that already exists rather than with inotify: this
     * thread wakes every 33 ms regardless, a stat() once a second is nothing
     * beside the work it does the rest of the time, and a poll cannot leak a
     * watch descriptor on a file that gets replaced by rename -- which is
     * exactly how supervisor writes this one.
     */
    auto last_conf_check = std::chrono::steady_clock::now();
    time_t last_conf_mtime = 0;
    {
        struct stat st;
        if (::stat(PRIVACY_BLUR_CONFIG_PATH, &st) == 0) last_conf_mtime = st.st_mtime;
    }

    while (predicting_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(predict_interval_ms_));

        if (!predicting_.load()) break;

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_conf_check).count()
            >= 1000) {
            last_conf_check = now;
            struct stat st;
            if (::stat(PRIVACY_BLUR_CONFIG_PATH, &st) == 0 && st.st_mtime != last_conf_mtime) {
                last_conf_mtime = st.st_mtime;
                PrivacyBlurConfig fresh;
                if (loadPrivacyBlurConfig(PRIVACY_BLUR_CONFIG_PATH, fresh, nullptr) &&
                    fresh.present) {
                    if (fresh.enabled != enabled_.load()) {
                        enabled_.store(fresh.enabled);
                        MA_LOGI(TAG, "masking %s (config changed)",
                                fresh.enabled ? "enabled" : "disabled");
                    }
                    if (fresh.alpha != cfg_.alpha) {
                        cfg_.alpha = fresh.alpha;
                        if (use_hw_pixelate_ && writeMaskAlpha(cfg_.alpha)) {
                            MA_LOGI(TAG, "mask opacity %d/255 (config changed)", cfg_.alpha);
                        }
                    }
                    /* Read per frame when the table is filled, so simply
                     * storing it is enough for the next frame to use it. */
                    cfg_.blocks_per_target = fresh.blocks_per_target;
                }
            }
        }

        if (!regions_inited_) continue;

        std::vector<BlurBox> predicted_boxes;
        {
            std::lock_guard<std::mutex> lock(tracker_mutex_);

            for (auto& tracker : trackers_) {
                tracker.predict(dt, process_noise_);
            }

            for (const auto& tracker : trackers_) {
                if (tracker.miss_count <= max_miss_) {
                    predicted_boxes.push_back(tracker.getBox());
                }
            }

            std::sort(predicted_boxes.begin(), predicted_boxes.end(),
                [](const BlurBox& a, const BlurBox& b) {
                    return a.score > b.score;
                });
        }

        /* Hide everything while switched off. The regions and the trackers stay
         * alive, so switching back on is one frame away and the video pipeline
         * is never disturbed. */
        if (!enabled_.load()) predicted_boxes.clear();

        /* No frame here by design: this thread only extrapolates position
         * between detections. Passing null tells applyRegions to move the tile
         * that is already uploaded rather than to re-render it -- re-rendering
         * would need pixels this thread must never touch. */
        applyRegions(predicted_boxes, nullptr);
    }

    MA_LOGI(TAG, "Prediction thread stopped");
}

// ============ Detection callback ============

void PrivacyBlur::onDetection(const std::vector<geometry::StreamBox>& detections,
                              const ma_img_t* frame) {
    if (!initialized_ || !regions_inited_) return;

    if (frames_seen_ < kIspSettleFrames) ++frames_seen_;

    /* Switched off means "conceal nothing", not "ignore this frame": feeding
     * the tracker an empty set is what makes the existing masks disappear. */
    std::vector<BlurBox> input;
    if (enabled_.load()) {
        /* The one place a StreamBox becomes the internal form. Same numbers,
         * same order; the type check has already happened at the call site. */
        input.reserve(detections.size());
        for (const auto& d : detections) {
            input.push_back({d.cx, d.cy, d.w, d.h, d.score});
        }
        truncateToCapacity(input);
    }

    std::vector<BlurBox> snapshot;
    {
        std::lock_guard<std::mutex> lock(tracker_mutex_);
        associateAndUpdate(input);
        for (const auto& t : trackers_) {
            if (t.miss_count <= max_miss_) snapshot.push_back(t.getBox());
        }
    }
    if (!enabled_.load()) snapshot.clear();

    /* The pixelating backend has to draw here and nowhere else: it averages the
     * pixels it is hiding, and this is the only place the frame exists. The
     * prediction thread keeps extrapolating position between detections, but it
     * cannot re-render -- it has no pixels, by design. */
    if ((use_overlayex_ || use_hw_pixelate_) && frame != nullptr) {
        std::sort(snapshot.begin(), snapshot.end(),
            [](const BlurBox& a, const BlurBox& b) { return a.score > b.score; });
        applyRegions(snapshot, frame);
    }
}

// ============ RGN hardware overlay management ============

void PrivacyBlur::initRegions() {
    if (regions_inited_) return;

    /*
     * Backend selection. The config file decides; the environment variables are
     * debug overrides only, kept because comparing backends on one device means
     * restarting a binary with a different variable rather than editing a file
     * the console also writes. They win over the config for exactly that
     * reason -- a developer's override should not be quietly undone by whatever
     * the console last saved.
     *
     * The three differ in what they can actually conceal: stock CV181x MOSAIC
     * fills its grid LUT from get_random_u32() and so renders as monochrome
     * static, COVEREX paints a solid rectangle (and only four of them per
     * channel), and the pixelating paths average the pixels they hide.
     */
    block_px_ = (cfg_.block_px == 8) ? 8 : 16;

    /*
     * Hardware colour mosaic when the kernel can do it, software compositing
     * when it cannot -- decided by probing, not by configuration.
     *
     * The two produce the same picture; they differ only in cost, by about
     * 38 ms and 3.6 MB of memory traffic per frame. Which of them a device can
     * run is a property of its kernel, not a preference its operator holds, so
     * asking the operator to pick would be asking them to know which driver
     * their firmware shipped with. Probe instead, and degrade quietly.
     *
     * The failure this protects against is real: on a stock driver the mosaic
     * grid is filled from get_random_u32(), so a hardware path that assumed
     * support would render television static rather than a mask.
     */
    const bool force_hw = (getenv("BLUR_HW_PIXELATE") != nullptr);
    use_hw_pixelate_ = false;
    { const char* t = getenv("BLUR_LUT_TEST"); lut_test_ = t ? atoi(t) : 0; }
    const bool want_coverex_early =
        (getenv("BLUR_COVEREX") != nullptr) || cfg_.backend == "coverex";
    const bool want_pixelate_early =
        (getenv("BLUR_PIXELATE") != nullptr) || force_hw || cfg_.backend == "pixelate";

    if (want_pixelate_early && !want_coverex_early) {
        rgn_fd_ = mosaic_lut_open();
        if (rgn_fd_ >= 0 && mosaic_lut_supported(rgn_fd_, vpss_grp_, vpss_chn_)) {
            use_hw_pixelate_ = true;
        } else {
            if (rgn_fd_ >= 0) {
                mosaic_lut_close(rgn_fd_);
                rgn_fd_ = -1;
            }
            MA_LOGI(TAG, "kernel has no mosaic colour table; using software compositing");
        }
    }

    const bool want_coverex =
        (getenv("BLUR_COVEREX") != nullptr) || cfg_.backend == "coverex";
    const bool want_pixelate =
        (getenv("BLUR_PIXELATE") != nullptr) || cfg_.backend == "pixelate";

    use_overlayex_ = !use_hw_pixelate_ && want_pixelate;
    use_coverex_   = !use_hw_pixelate_ && !use_overlayex_ && want_coverex;

    /* Debug override for the block size, wider than the config allows because
     * the software pixelate path is not bound by the hardware's 8/16 grid and
     * tuning it is exactly what this variable is for. */
    if (const char* b = getenv("BLUR_BLOCK_PX")) {
        const int v = atoi(b);
        if (v >= 4 && v <= 128) block_px_ = v;
    }

    /* Capacity is how many detections may be concealed at once; region_count_
     * is how many RGN regions that takes. They differ only for the software
     * pixelate path, which composites every mask into one region.
     *
     * One full-frame region, not one per detection, because regions on a VPSS
     * layer may not overlap (rgn.c:296) and the overlap test uses the canvas
     * size fixed at Create -- not the bitmap actually drawn. A per-detection
     * region would therefore need every mask to sit a full canvas width apart,
     * which no real scene obliges. Compositing sidesteps the overlap rule and
     * the region-count limit at once, at the cost of one full-frame buffer. */
    capacity_ = cfg_.max_regions;
    if (capacity_ < 1) capacity_ = 1;
    if (capacity_ > kMaxRegionsLimit) capacity_ = kMaxRegionsLimit;
    if (use_coverex_ && capacity_ > kMaxCoverexRegions) capacity_ = kMaxCoverexRegions;
    region_count_ = use_overlayex_ ? 1 : capacity_;

    MA_LOGI(TAG, "blur backend: %s (capacity=%d, regions=%d, block=%dpx)",
            use_hw_pixelate_ ? "MOSAIC+LUT (hardware pixelate)"
                             : use_overlayex_ ? "OVERLAY (software pixelate)"
                           : (use_coverex_ ? "COVEREX (solid)" : "MOSAIC (noise)"),
            capacity_, region_count_, block_px_);

    handles_.clear();
    tile_uploaded_.assign(region_count_, false);

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    for (int i = 0; i < region_count_; i++) {
        RGN_HANDLE hRgn = kRgnHandleBase + i;

        RGN_ATTR_S stRgnAttr;
        memset(&stRgnAttr, 0, sizeof(stRgnAttr));
        stRgnAttr.enType = use_overlayex_ ? OVERLAY_RGN
                                          : (use_coverex_ ? COVEREX_RGN : MOSAIC_RGN);
        if (use_overlayex_) {
            /* OVERLAY rather than OVERLAYEX: the two carry identical bitmap
             * attributes, but rgn.c:1193 refuses to create an OVERLAYEX region
             * unless VPSS is running in RGNEX mode, and that mode reserves
             * enough ION up front to have evicted a model the last time it was
             * tried. Plain OVERLAY has no such precondition.
             *
             * One canvas sized for the largest tile we will ever draw. The
             * driver row-copies whatever smaller bitmap is handed to SetBitMap,
             * so a generous canvas costs ION once instead of a resize per
             * detection. Two canvases so a bitmap update never tears against
             * the compositor reading the previous one. */
            stRgnAttr.unAttr.stOverlay.enPixelFormat   = PIXEL_FORMAT_ARGB_8888;
            stRgnAttr.unAttr.stOverlay.stSize.u32Width  = (CVI_U32)stream_width_;
            stRgnAttr.unAttr.stOverlay.stSize.u32Height = (CVI_U32)stream_height_;
            /* One canvas: two would double a 3.6 MB allocation for tear-free
             * updates nobody would notice on a privacy mask. */
            stRgnAttr.unAttr.stOverlay.u32CanvasNum = 1;
            stRgnAttr.unAttr.stOverlay.u32BgColor   = 0;
        }

        CVI_S32 ret = CVI_RGN_Create(hRgn, &stRgnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGE(TAG, "CVI_RGN_Create(%d) failed: 0x%x", hRgn, ret);
            continue;
        }

        RGN_CHN_ATTR_S stChnAttr;
        memset(&stChnAttr, 0, sizeof(stChnAttr));
        stChnAttr.bShow  = CVI_FALSE;
        stChnAttr.enType = use_overlayex_ ? OVERLAY_RGN
                                          : (use_coverex_ ? COVEREX_RGN : MOSAIC_RGN);
        if (use_overlayex_) {
            stChnAttr.unChnAttr.stOverlayChn.stPoint.s32X = 0;
            stChnAttr.unChnAttr.stOverlayChn.stPoint.s32Y = 0;
            /* Layer, not slot. VPSS has RGN_MAX_LAYER_VPSS == 2 overlay layers
             * holding RGN_MAX_NUM_VPSS == 8 regions each, so numbering layers
             * by region index makes every region past the second fail attach
             * with NOT_PERM. The masks never overlap, so they can all share
             * one layer. */
            stChnAttr.unChnAttr.stOverlayChn.u32Layer     = 0;
            ret = CVI_RGN_AttachToChn(hRgn, &stChn, &stChnAttr);
            if (ret != CVI_SUCCESS) {
                MA_LOGE(TAG, "CVI_RGN_AttachToChn(%d) failed: 0x%x", hRgn, ret);
                CVI_RGN_Destroy(hRgn);
                continue;
            }
            handles_.push_back(hRgn);
            continue;
        }
        if (use_coverex_) {
            stChnAttr.unChnAttr.stCoverExChn.enCoverType = AREA_RECT;
            stChnAttr.unChnAttr.stCoverExChn.stRect      = RECT_S{0, 0, 64, 64};
            stChnAttr.unChnAttr.stCoverExChn.u32Color    = 0x00202020;
            stChnAttr.unChnAttr.stCoverExChn.u32Layer    = i;
            ret = CVI_RGN_AttachToChn(hRgn, &stChn, &stChnAttr);
            if (ret != CVI_SUCCESS) {
                MA_LOGE(TAG, "CVI_RGN_AttachToChn(%d) failed: 0x%x", hRgn, ret);
                CVI_RGN_Destroy(hRgn);
                continue;
            }
            handles_.push_back(hRgn);
            continue;
        }
        stChnAttr.unChnAttr.stMosaicChn.stRect.s32X      = 0;
        stChnAttr.unChnAttr.stMosaicChn.stRect.s32Y      = 0;
        stChnAttr.unChnAttr.stMosaicChn.stRect.u32Width   = 64;
        stChnAttr.unChnAttr.stMosaicChn.stRect.u32Height  = 64;
        stChnAttr.unChnAttr.stMosaicChn.enBlkSize         =
            (block_px_ == 8) ? MOSAIC_BLK_SIZE_8 : MOSAIC_BLK_SIZE_16;
        stChnAttr.unChnAttr.stMosaicChn.u32Layer          = i;

        ret = CVI_RGN_AttachToChn(hRgn, &stChn, &stChnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGE(TAG, "CVI_RGN_AttachToChn(%d) failed: 0x%x", hRgn, ret);
            CVI_RGN_Destroy(hRgn);
            continue;
        }

        handles_.push_back(hRgn);
    }

    if (handles_.empty()) {
        MA_LOGE(TAG, "Failed to create any mosaic regions, disabling blur");
        return;
    }

    regions_inited_ = true;
    MA_LOGI(TAG, "Initialized %d/%d blur regions on VPSS(%d,%d)",
            (int)handles_.size(), region_count_, vpss_grp_, vpss_chn_);
}

void PrivacyBlur::deinitRegions() {
    if (!regions_inited_ && handles_.empty()) return;

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    for (auto hRgn : handles_) {
        CVI_RGN_DetachFromChn(hRgn, &stChn);
        CVI_RGN_Destroy(hRgn);
    }

    handles_.clear();
    regions_inited_ = false;

    /* Closed here rather than left to process exit, because an application may
     * tear the blur down and bring it back up (the supervisor switches apps in
     * place) and leaking one fd per cycle would eventually exhaust the table. */
    if (rgn_fd_ >= 0) {
        mosaic_lut_close(rgn_fd_);
        rgn_fd_ = -1;
    }

    MA_LOGI(TAG, "Deinitialized blur regions");
}

// Apply predicted/tracked boxes to RGN hardware overlays
/*
 * Pixelate: average the source pixels under each block and paint that average
 * back as a flat square. This is what the reference privacy mosaic does, and
 * why it reads as "the scene, coarsened" rather than as a coloured rectangle --
 * the tile keeps the subject's overall luminance and hue while destroying every
 * feature smaller than a block.
 *
 * Written by hand rather than with OpenCV: a box average and a nearest-neighbour
 * expand are about forty lines, and adding an OpenCV dependency to this solution
 * to get them would cost far more than it saves.
 */
/*
 * Composite every mask into one frame-sized ARGB bitmap and upload it.
 *
 * Pixelate: average the source pixels under each block and paint that average
 * back as a flat square. That is what makes the result read as "the scene,
 * coarsened" rather than as a coloured rectangle -- the mask keeps the
 * subject's luminance and hue while destroying every feature smaller than a
 * block. Hardware MOSAIC cannot do this (the CV181x driver fills its grid from
 * get_random_u32(), so it renders as static) and COVEREX only paints flat
 * colour.
 *
 * The averages come from the inference frame while the mask is drawn at stream
 * resolution, so the two coordinate spaces are converted separately: box ->
 * source pixels uses the frame's own geometry, box -> destination uses the
 * letterbox mapping the caller computed.
 *
 * Written by hand rather than with OpenCV: a box average and a nearest expand
 * are forty lines, and taking an OpenCV dependency for them would cost this
 * solution far more than it saves.
 */
bool PrivacyBlur::renderPixelated(const std::vector<BlurBox>& boxes,
                                 int active_count, const ma_img_t* frame,
                                 int target_w, int target_h,
                                 int offset_x, int offset_y) {
    if (frame == nullptr || frame->data == nullptr) return false;
    if (handles_.empty()) return false;

    const int fw = frame->width;
    const int fh = frame->height;
    if (fw <= 0 || fh <= 0) return false;

    const size_t px = (size_t)stream_width_ * stream_height_;
    if (tile_.size() != px) tile_.assign(px, 0);
    else std::fill(tile_.begin(), tile_.end(), 0u);

    const uint8_t* src = (const uint8_t*)frame->data;
    bool any = false;

    for (int k = 0; k < active_count && k < (int)boxes.size(); k++) {
        const auto& box = boxes[k];
        if (box.w > 0.7f && box.h > 0.7f) continue;  // ISP noise frame guard

        /* Source rect, in the inference frame the averages come from. */
        int sx = (int)((box.x - box.w / 2.0f) * fw);
        int sy = (int)((box.y - box.h / 2.0f) * fh);
        int sw = (int)(box.w * fw);
        int sh = (int)(box.h * fh);
        if (sx < 0) { sw += sx; sx = 0; }
        if (sy < 0) { sh += sy; sy = 0; }
        if (sx + sw > fw) sw = fw - sx;
        if (sy + sh > fh) sh = fh - sy;
        if (sw <= 0 || sh <= 0) continue;

        /* Destination rect, in stream pixels. */
        int dx = (int)((box.x - box.w / 2.0f) * target_w + offset_x);
        int dy = (int)((box.y - box.h / 2.0f) * target_h + offset_y);
        int dw = (int)(box.w * target_w);
        int dh = (int)(box.h * target_h);
        if (dx < 0) { dw += dx; dx = 0; }
        if (dy < 0) { dh += dy; dy = 0; }
        if (dx + dw > stream_width_)  dw = stream_width_ - dx;
        if (dy + dh > stream_height_) dh = stream_height_ - dy;
        if (dw <= 0 || dh <= 0) continue;

        const int bx_n = std::max(1, dw / block_px_);
        const int by_n = std::max(1, dh / block_px_);

        for (int by = 0; by < by_n; by++) {
            for (int bx = 0; bx < bx_n; bx++) {
                const int x0 = sx + sw * bx / bx_n, x1 = sx + sw * (bx + 1) / bx_n;
                const int y0 = sy + sh * by / by_n, y1 = sy + sh * (by + 1) / by_n;

                uint32_t r = 0, g = 0, b = 0, n = 0;
                for (int y = y0; y < y1; y++) {
                    const uint8_t* row = src + (size_t)y * fw * 3;
                    for (int x = x0; x < x1; x++) {
                        r += row[x * 3 + 0]; g += row[x * 3 + 1]; b += row[x * 3 + 2]; n++;
                    }
                }
                if (n == 0) continue;
                const uint32_t argb = 0xFF000000u | ((r / n) << 16) | ((g / n) << 8) | (b / n);

                const int ex0 = dx + dw * bx / bx_n, ex1 = dx + dw * (bx + 1) / bx_n;
                const int ey0 = dy + dh * by / by_n, ey1 = dy + dh * (by + 1) / by_n;
                for (int y = ey0; y < ey1; y++) {
                    uint32_t* drow = tile_.data() + (size_t)y * stream_width_;
                    for (int x = ex0; x < ex1; x++) drow[x] = argb;
                }
            }
        }
        any = true;
    }

    BITMAP_S bmp;
    memset(&bmp, 0, sizeof(bmp));
    bmp.enPixelFormat = PIXEL_FORMAT_ARGB_8888;
    bmp.u32Width  = (CVI_U32)stream_width_;
    bmp.u32Height = (CVI_U32)stream_height_;
    bmp.pData     = tile_.data();

    const CVI_S32 ret = CVI_RGN_SetBitMap(handles_[0], &bmp);
    if (ret != CVI_SUCCESS) {
        /* Keep whatever is already displayed: a stale mosaic still conceals,
         * whereas dropping the region would briefly expose the subject. */
        MA_LOGW(TAG, "CVI_RGN_SetBitMap failed: 0x%x", ret);
        return false;
    }
    return any;
}

void PrivacyBlur::applyRegions(const std::vector<BlurBox>& boxes,
                              const ma_img_t* frame) {
    if (!regions_inited_ || stream_width_ <= 0 || stream_height_ <= 0) return;

    /*
     * Normalised [0,1] -> stream pixels, and nothing else.
     *
     * This used to undo a letterbox: it scaled the vertical axis by
     * width/height and shifted by a negative offset, on the assumption that
     * callers hand over coordinates normalised against a square model input
     * with the picture letterboxed inside it. No such letterbox exists. The
     * preprocessing step that feeds the model, rgb888_to_rgb888_planar() in
     * ma_cv.cpp, derives its horizontal and vertical steps independently
     * (beta_w = sw/dw, beta_h = sh/dh) and pads nothing -- it stretches. A
     * detection therefore maps back linearly to the frame it was run on.
     *
     * Compensating for padding that was never added put the mask 16/9 too tall
     * on a 1280x720 stream. Measured, with the probe feeding boxes of known
     * size: a box asking for 72, 180 and 360 pixels of height rendered 128, 320
     * and 640 -- exactly h_norm * stream_width in each case, because
     * target_h had been scaled up to equal stream_width.
     *
     * Callers whose coordinates are normalised against a frame of a different
     * shape (a 4:3 inference channel against a 16:9 stream, say) must convert
     * before calling. That belongs to them: only the caller knows what its
     * detector was run on, and guessing here is what produced this bug.
     */
    const int target_w     = stream_width_;
    const int target_h     = stream_height_;
    const int32_t offset_x = 0;
    const int32_t offset_y = 0;

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    int num_handles = (int)handles_.size();
    int active_count = std::min((int)boxes.size(), num_handles);

    /* Pixelating backend: one region holds the whole composite, so the
     * per-handle loop below does not apply. Everything visible is decided by
     * the bitmap; the region itself is simply pinned at the origin. */
    if (use_overlayex_) {
        RGN_CHN_ATTR_S a;
        memset(&a, 0, sizeof(a));
        a.enType = OVERLAY_RGN;
        a.unChnAttr.stOverlayChn.stPoint.s32X = 0;
        a.unChnAttr.stOverlayChn.stPoint.s32Y = 0;
        a.unChnAttr.stOverlayChn.u32Layer     = 0;

        const int n = std::min((int)boxes.size(), capacity_);
        const bool drawn = (frame != nullptr)
            && renderPixelated(boxes, n, frame, target_w, target_h, offset_x, offset_y);
        if (frame != nullptr) tile_uploaded_[0] = drawn;

        a.bShow = tile_uploaded_[0] ? CVI_TRUE : CVI_FALSE;
        CVI_RGN_SetDisplayAttr(handles_[0], &stChn, &a);
        return;
    }

    for (int i = 0; i < num_handles; i++) {
        RGN_CHN_ATTR_S stChnAttr;
        memset(&stChnAttr, 0, sizeof(stChnAttr));
        stChnAttr.enType = use_coverex_ ? COVEREX_RGN : MOSAIC_RGN;

        /* Fill whichever member of the union the region type actually reads.
         * Writing the MOSAIC member while the region is COVEREX is not a
         * compile error and not a runtime error either -- it just draws the
         * cover somewhere else, which is the worst kind of wrong for a privacy
         * mask. */
        auto set_rect = [&](bool show, int x, int y, int w, int h) {
            stChnAttr.bShow = show ? CVI_TRUE : CVI_FALSE;
            if (use_overlayex_) {
                /* OVERLAYEX carries only a position; the extent comes from the
                 * bitmap that was uploaded, which is why the tile has to be
                 * rendered before this is called. */
                stChnAttr.unChnAttr.stOverlayChn.stPoint.s32X = x;
                stChnAttr.unChnAttr.stOverlayChn.stPoint.s32Y = y;
                stChnAttr.unChnAttr.stOverlayChn.u32Layer     = 0;
            } else if (use_coverex_) {
                stChnAttr.unChnAttr.stCoverExChn.enCoverType = AREA_RECT;
                stChnAttr.unChnAttr.stCoverExChn.stRect      = RECT_S{x, y, (CVI_U32)w, (CVI_U32)h};
                stChnAttr.unChnAttr.stCoverExChn.u32Color    = 0x00202020;
                stChnAttr.unChnAttr.stCoverExChn.u32Layer    = i;
            } else {
                stChnAttr.unChnAttr.stMosaicChn.stRect.s32X     = x;
                stChnAttr.unChnAttr.stMosaicChn.stRect.s32Y     = y;
                stChnAttr.unChnAttr.stMosaicChn.stRect.u32Width  = (CVI_U32)w;
                stChnAttr.unChnAttr.stMosaicChn.stRect.u32Height = (CVI_U32)h;
                stChnAttr.unChnAttr.stMosaicChn.enBlkSize        = MOSAIC_BLK_SIZE_16;
                stChnAttr.unChnAttr.stMosaicChn.u32Layer         = i;
            }
        };

        if (i < active_count) {
            const auto& box = boxes[i];

            /*
             * Drop near-full-frame boxes only while the ISP is still settling.
             *
             * The guard exists because the noise frames produced during ISP
             * init detect as one enormous box and handing that to VPSS could
             * crash it. But "box covers most of the frame" is a proxy for "this
             * is an init artefact", and it is a bad one: a person standing
             * close to the camera produces exactly the same shape. Applied
             * forever, it meant the nearer someone was -- the case where they
             * are most identifiable -- the more certainly they went unmasked.
             * That is a privacy feature failing in the direction that matters.
             *
             * So keep the protection where the danger actually is, the first
             * frames after start-up, and outside that window let the clamping
             * below handle an oversized box the ordinary way. A mask trimmed to
             * the frame is correct; no mask at all is not.
             */
            if (box.w > 0.7f && box.h > 0.7f) {
                if (frames_seen_ < kIspSettleFrames) {
                    set_rect(false, 0, 0, 64, 64);
                    CVI_RGN_SetDisplayAttr(handles_[i], &stChn, &stChnAttr);
                    continue;
                }
                if (!large_box_warned_) {
                    large_box_warned_ = true;
                    MA_LOGI(TAG, "masking a near-full-frame subject (%.2fx%.2f); "
                                 "clamping to the frame instead of skipping", box.w, box.h);
                }
            }

            int left = (int)((box.x - box.w / 2.0f) * target_w + offset_x);
            int top  = (int)((box.y - box.h / 2.0f) * target_h + offset_y);
            int w    = (int)(box.w * target_w);
            int h    = (int)(box.h * target_h);

            // Clamp to frame bounds
            left = std::max(0, left);
            top  = std::max(0, top);
            w    = std::min(w, stream_width_ - left);
            h    = std::min(h, stream_height_ - top);

            // Align to 8 pixels (MOSAIC hardware requirement)
            left = left & ~7;
            top  = top & ~7;
            w    = std::max(8, (w + 7) & ~7);
            h    = std::max(8, (h + 7) & ~7);

            // Re-check bounds after alignment
            if (left + w > stream_width_) w = (stream_width_ - left) & ~7;
            if (top + h > stream_height_) h = (stream_height_ - top) & ~7;
            w = std::max(8, w);
            h = std::max(8, h);

            set_rect(true, left, top, w, h);
        } else {
            set_rect(false, 0, 0, 64, 64);
        }

        CVI_S32 ret = CVI_RGN_SetDisplayAttr(handles_[i], &stChn, &stChnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGW(TAG, "CVI_RGN_SetDisplayAttr(%d) failed: 0x%x", handles_[i], ret);
        }
    }

    /* After the display attributes, never before: the driver derives the grid
     * from the regions currently shown, so querying earlier would describe the
     * previous frame's geometry. */
    if (use_hw_pixelate_ && frame != nullptr) {
        updateHwLut(frame, boxes, target_w, target_h, offset_x, offset_y);
    } else if (use_hw_pixelate_ && !lut_.empty()) {
        /*
         * Moving the regions invalidates the colour table, so re-hand it over.
         *
         * The driver only honours a stored table whose stride and grid
         * dimensions match the geometry it is about to render, and it derives
         * that geometry from wherever the regions are now. This thread has just
         * moved them, so the table uploaded at the last detection no longer
         * matches and the driver discards it -- falling back to a constant
         * fill, which renders as a flat slab of RGB332 index 1. That is the
         * translucent navy rectangle that appeared over a moving face while a
         * stationary one was masked properly.
         *
         * Re-uploading the same bytes fixes it because the upload path
         * recomputes the geometry and stores the table against it. The colours
         * are then one prediction tick stale -- at most 33 ms of movement,
         * across cells 16 px wide -- which is invisible next to the difference
         * between a mosaic and no mosaic at all. No pixels are read here, so
         * this stays within what the prediction thread is allowed to do.
         */
        mosaic_lut_apply(rgn_fd_, vpss_grp_, vpss_chn_, lut_.data(), (uint32_t)lut_.size());
    }
}

/*
 * One RGB332 byte per grid cell, averaged from the pixels that cell will hide.
 *
 * The cell rectangle is in stream coordinates because that is the space the
 * scaler masks in, while the averages have to come from the inference frame --
 * the only image the application holds. The two are related by the same
 * letterbox mapping used to place the regions, applied in reverse.
 */

/*
 * Pack a colour into the RGB332 byte the privacy-mask unit expects.
 *
 * Two things make the naive shift wrong, and both were visible on hardware:
 *
 * Truncating instead of rounding throws away up to 31 levels per channel and
 * always downward, so the mask came out markedly darker than what it covered
 * and anything below 32 collapsed to 0 -- which the driver clamps to 1 and
 * renders black.
 *
 * Rounding each channel independently then breaks neutral colours, because red
 * and green get 8 levels while blue gets 4: a grey of (112,110,109) lands on
 * red level 4, green level 3 and blue level 2, i.e. (128,96,128) -- a magenta
 * cast on what should stay grey, and a navy cast on near-black. So a colour
 * close to neutral is quantised from its luma once and the same level is used
 * for all three channels, which keeps greys grey at the cost of nothing: a
 * neutral colour has no hue to lose.
 */
static inline uint8_t rgb_to_rgb332(uint8_t r, uint8_t g, uint8_t b)
{
    const int mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
    const int mn = r < g ? (r < b ? r : b) : (g < b ? g : b);

    uint32_t rv, gv, bv;
    if (mx - mn <= 24) {
        /*
         * Near-neutral. Red and green land on multiples of 32 and blue on
         * multiples of 64, so the only outputs the three channels can agree on
         * are 0, 64, 128 and 192. Snapping a grey to the nearest of those keeps
         * it grey; letting each channel round independently is what turned a
         * mid grey magenta and a near-black navy. Four levels is a real loss of
         * detail, but a neutral colour has no hue to spend on hiding it.
         */
        const int y = (r * 77 + g * 151 + b * 28) >> 8;
        const uint32_t lvl = (uint32_t)((y + 32) / 64);  /* 0..3 -> 0,64,128,192 */
        bv = lvl;
        rv = gv = lvl * 2;
    } else {
        rv = ((uint32_t)r + 16) / 32;
        gv = ((uint32_t)g + 16) / 32;
        bv = ((uint32_t)b + 32) / 64;
    }
    if (rv > 7) rv = 7;
    if (gv > 7) gv = 7;
    if (bv > 3) bv = 3;
    return (uint8_t)((rv << 5) | (gv << 2) | bv);
}

bool PrivacyBlur::updateHwLut(const ma_img_t* frame, const std::vector<BlurBox>& boxes,
                             int target_w, int target_h,
                             int offset_x, int offset_y) {
    if (rgn_fd_ < 0 || frame == nullptr || frame->data == nullptr) return false;

    rgn_mosaic_lut_u layout;
    if (mosaic_lut_query(rgn_fd_, vpss_grp_, vpss_chn_, &layout) != 0) return false;
    if (layout.grid_w == 0 || layout.grid_h == 0 || layout.stride == 0) return false;

    const size_t need = (size_t)layout.stride * layout.grid_h;
    if (lut_.size() != need) lut_.assign(need, 0);
    else std::fill(lut_.begin(), lut_.end(), 0u);
    /* Per-cell averages kept unquantised: the grouping pass averages these
     * again, and averaging RGB332 values would compound the quantisation error
     * of every cell into the group's colour. */
    if (cell_rgb_.size() != need * 3) cell_rgb_.assign(need * 3, 0);

    const int fw = frame->width, fh = frame->height;
    if (fw <= 0 || fh <= 0) return false;
    const uint8_t* src = (const uint8_t*)frame->data;
    const int g = layout.grid_size;

    if (lut_test_ == 2) {
        uint64_t R = 0, G = 0, B = 0, N = 0;
        for (int y = 0; y < fh; y += 4) {
            const uint8_t* row = src + (size_t)y * fw * 3;
            for (int x = 0; x < fw; x += 4) {
                R += row[x * 3 + 0]; G += row[x * 3 + 1]; B += row[x * 3 + 2]; N++;
            }
        }
        if (N) {
            frame_avg_ = rgb_to_rgb332((uint8_t)(R/N), (uint8_t)(G/N), (uint8_t)(B/N));
            MA_LOGI(TAG, "frame avg rgb=(%llu,%llu,%llu) rgb332=0x%02x",
                    (unsigned long long)(R/N), (unsigned long long)(G/N),
                    (unsigned long long)(B/N), frame_avg_);
        }
    }

    for (int gy = 0; gy < layout.grid_h; gy++) {
        for (int gx = 0; gx < layout.grid_w; gx++) {
            /* Cell bounds in stream pixels, then back through the letterbox
             * mapping into the inference frame. */
            const int sx0 = layout.start_x + gx * g;
            const int sy0 = layout.start_y + gy * g;
            int x0 = (int)((float)(sx0 - offset_x) / target_w * fw);
            int x1 = (int)((float)(sx0 + g - offset_x) / target_w * fw);
            int y0 = (int)((float)(sy0 - offset_y) / target_h * fh);
            int y1 = (int)((float)(sy0 + g - offset_y) / target_h * fh);
            /*
             * Clamp the sampling rectangle to the nearest valid row and column
             * rather than abandoning a cell that hangs over the edge of the
             * frame.
             *
             * Leaving a cell uncoloured is not the same as leaving it alone.
             * The mask unit runs with force_alpha set, so every cell inside the
             * masked rectangle is composited whatever its index, and index 0 is
             * RGB332 black. "No colour was computed for this cell" and "this
             * cell is black" would therefore be the same state, and the second
             * one is visible. Clamping guarantees every cell has a colour taken
             * from real pixels, so that state cannot arise.
             */
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            if (x1 > fw) x1 = fw;
            if (y1 > fh) y1 = fh;
            if (x0 >= fw) x0 = fw - 1;
            if (y0 >= fh) y0 = fh - 1;
            if (x1 <= x0) x1 = x0 + 1;
            if (y1 <= y0) y1 = y0 + 1;

            const size_t idx = (size_t)gy * layout.stride + gx;
            uint32_t r = 0, gg = 0, b = 0, n = 0;
            for (int y = y0; y < y1; y++) {
                const uint8_t* row = src + (size_t)y * fw * 3;
                for (int x = x0; x < x1; x++) {
                    r += row[x * 3 + 0]; gg += row[x * 3 + 1]; b += row[x * 3 + 2]; n++;
                }
            }
            /* RGB332: 3 bits red, 3 green, 2 blue. */
            uint8_t v;
            if (n == 0) {
                /*
                 * The clamp above makes this unreachable, and it is kept only
                 * so that a future change to the mapping cannot silently
                 * reintroduce a black cell. Borrow the neighbour that was just
                 * computed -- the cell to the left, or the one above at the
                 * start of a row -- because an adjacent cell covers adjacent
                 * pixels and so is the closest thing to the right answer that
                 * is available without sampling.
                 */
                v = (gx > 0)   ? lut_[(size_t)gy * layout.stride + (gx - 1)]
                  : (gy > 0)   ? lut_[(size_t)(gy - 1) * layout.stride + gx]
                               : 0u;
            } else if (lut_test_) {
                /* Known-pattern probe. The byte layout the hardware expects is
                 * not documented and the observed colours did not match the
                 * scene, so rather than guess between a channel-order bug and a
                 * colour-space one, paint three vertical bands of pure red,
                 * pure green and pure blue under this packing and read the
                 * answer off the stream. */
                if (lut_test_ == 1) {
                    const int band = gx * 3 / (layout.grid_w ? layout.grid_w : 1);
                    v = (band == 0) ? 0xE0u : (band == 1) ? 0x1Cu : 0x03u;
                } else {
                    /* Whole-frame average painted into every cell. If the mask
                     * comes out matching the room's overall brightness then the
                     * pixel reading and the RGB332 packing are both fine and any
                     * darkness is coming from where the per-cell rectangles
                     * sample, not from how they are averaged. */
                    v = frame_avg_;
                }
            } else {
                /*
                 * Round to the nearest representable level rather than
                 * truncating. RGB332 gives red and green 8 steps of 32 and
                 * blue only 4 steps of 64, so a plain shift throws away up to
                 * 31 levels per channel -- always downward. That reads as a
                 * mask noticeably darker than what it covers, with a yellow
                 * cast because blue loses twice as much, and it crushes
                 * anything below 32 to 0, which the driver then clamps to 1
                 * and renders black.
                 */
                v = rgb_to_rgb332((uint8_t)(r / n), (uint8_t)(gg / n), (uint8_t)(b / n));
            }
            cell_rgb_[idx * 3 + 0] = (n == 0) ? 0 : (uint8_t)(r / n);
            cell_rgb_[idx * 3 + 1] = (n == 0) ? 0 : (uint8_t)(gg / n);
            cell_rgb_[idx * 3 + 2] = (n == 0) ? 0 : (uint8_t)(b / n);
            lut_[idx] = v ? v : 1;
        }
    }

    /*
     * Second pass: coarsen each subject to at most blocks_per_target blocks.
     *
     * The hardware cell stays 16 px -- it has no other size to offer -- so the
     * bigger block is made by handing a kxk group of cells one shared colour.
     * Averaging the per-cell averages rather than re-reading the frame keeps
     * this to a few thousand additions over a grid that is at most 80x45, and
     * the result is identical as long as the cells are the same size, which
     * they are everywhere except at a clamped frame edge.
     *
     * Done per box, so two subjects at different distances each get the
     * coarseness their own size calls for, and cells belonging to no box keep
     * the per-cell colour computed above.
     */
    if (cfg_.blocks_per_target > 0) {
        for (const auto& box : boxes) {
            const int bx0 = (int)((box.x - box.w * 0.5f) * target_w + offset_x);
            const int by0 = (int)((box.y - box.h * 0.5f) * target_h + offset_y);
            const int bw  = (int)(box.w * target_w);
            const int bh  = (int)(box.h * target_h);
            if (bw <= 0 || bh <= 0) continue;

            /* Cells this box spans, clipped to the grid. */
            int cx0 = (bx0 - layout.start_x) / g;
            int cy0 = (by0 - layout.start_y) / g;
            int cx1 = (bx0 + bw - layout.start_x + g - 1) / g;
            int cy1 = (by0 + bh - layout.start_y + g - 1) / g;
            if (cx0 < 0) cx0 = 0;
            if (cy0 < 0) cy0 = 0;
            if (cx1 > layout.grid_w) cx1 = layout.grid_w;
            if (cy1 > layout.grid_h) cy1 = layout.grid_h;
            if (cx1 - cx0 < 1 || cy1 - cy0 < 1) continue;

            /* How many cells may span the subject, and hence how many of them
             * share a colour. Driven by the longer edge so a tall narrow box is
             * coarsened as much as a wide one. */
            const int cells_across = std::max(cx1 - cx0, cy1 - cy0);
            int k = (cells_across + cfg_.blocks_per_target - 1) / cfg_.blocks_per_target;
            if (k < 1) k = 1;
            if (k == 1) continue;  /* already coarse enough */

            /* Group boundaries are anchored to the box, not to the grid: a
             * group half outside the box would pull in colours from outside
             * the subject and smear the mask past its own edge. */
            for (int cy = cy0; cy < cy1; cy += k) {
                for (int cx = cx0; cx < cx1; cx += k) {
                    const int gy1 = std::min(cy + k, cy1);
                    const int gx1 = std::min(cx + k, cx1);
                    uint32_t r = 0, gg = 0, b = 0, n = 0;
                    for (int yy = cy; yy < gy1; ++yy) {
                        for (int xx = cx; xx < gx1; ++xx) {
                            const size_t id = (size_t)yy * layout.stride + xx;
                            r += cell_rgb_[id * 3 + 0];
                            gg += cell_rgb_[id * 3 + 1];
                            b += cell_rgb_[id * 3 + 2];
                            ++n;
                        }
                    }
                    if (n == 0) continue;
                    uint8_t v = rgb_to_rgb332((uint8_t)(r / n), (uint8_t)(gg / n),
                                              (uint8_t)(b / n));
                    if (!v) v = 1;
                    for (int yy = cy; yy < gy1; ++yy) {
                        for (int xx = cx; xx < gx1; ++xx) {
                            lut_[(size_t)yy * layout.stride + xx] = v;
                        }
                    }
                }
            }
        }
    }

    return mosaic_lut_apply(rgn_fd_, vpss_grp_, vpss_chn_,
                            lut_.data(), (uint32_t)lut_.size()) == 0;
}

void pixelateRgb888(void* rgb888, int width, int height,
                    const std::vector<geometry::InferBox>& boxes, int block_px) {
    if (rgb888 == nullptr || width <= 0 || height <= 0 || boxes.empty()) return;
    if (block_px < 2) block_px = 2;

    uint8_t* buf = static_cast<uint8_t*>(rgb888);

    for (const auto& b : boxes) {
        /* Centre form to edges, then clamp. A detection can legitimately run
         * past the frame when the subject is half out of shot, and the part
         * that is still visible is exactly the part that has to be covered --
         * so clamp rather than skip. */
        int left   = (int)(b.left() * width);
        int top    = (int)(b.top() * height);
        int right  = (int)(b.right() * width);
        int bottom = (int)(b.bottom() * height);
        if (left < 0) left = 0;
        if (top < 0) top = 0;
        if (right > width) right = width;
        if (bottom > height) bottom = height;
        if (right - left < 1 || bottom - top < 1) continue;

        /* Snap the block grid to the frame, not to the box, so two overlapping
         * detections agree on where the blocks fall and their overlap does not
         * come out as a seam of half-sized cells. */
        for (int by = (top / block_px) * block_px; by < bottom; by += block_px) {
            const int y0 = by < top ? top : by;
            const int y1 = (by + block_px > bottom) ? bottom : by + block_px;
            if (y1 <= y0) continue;

            for (int bx = (left / block_px) * block_px; bx < right; bx += block_px) {
                const int x0 = bx < left ? left : bx;
                const int x1 = (bx + block_px > right) ? right : bx + block_px;
                if (x1 <= x0) continue;

                uint32_t r = 0, g = 0, bl = 0, n = 0;
                for (int y = y0; y < y1; ++y) {
                    const uint8_t* row = buf + ((size_t)y * width + x0) * 3;
                    for (int x = x0; x < x1; ++x, row += 3) {
                        r += row[0];
                        g += row[1];
                        bl += row[2];
                        ++n;
                    }
                }
                if (n == 0) continue;
                const uint8_t ar = (uint8_t)(r / n);
                const uint8_t ag = (uint8_t)(g / n);
                const uint8_t ab = (uint8_t)(bl / n);

                for (int y = y0; y < y1; ++y) {
                    uint8_t* row = buf + ((size_t)y * width + x0) * 3;
                    for (int x = x0; x < x1; ++x, row += 3) {
                        row[0] = ar;
                        row[1] = ag;
                        row[2] = ab;
                    }
                }
            }
        }
    }
}

}  // namespace privacy_blur
