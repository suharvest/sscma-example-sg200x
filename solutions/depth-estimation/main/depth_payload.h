#ifndef _DEPTH_PAYLOAD_H_
#define _DEPTH_PAYLOAD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "depth_estimator.h"

namespace depth {

/*
 * Frame statistics over the depth map.
 *
 * Every value here is relative. The model returns depth in its own units with
 * no absolute scale, so nothing in this struct, in the JSON it produces, or in
 * the documentation describes a distance.
 */
struct DepthStats {
    /* Raw model units, smaller = nearer. */
    float min  = 0.0f;
    float max  = 0.0f;
    float mean = 0.0f;
    float p02  = 0.0f;
    float p50  = 0.0f;
    float p98  = 0.0f;

    /* Fraction of pixels whose proximity is at or above the near threshold. */
    float near_ratio   = 0.0f;
    bool  near_present = false;

    /* 3x3 grid, row-major (index 4 is the centre). Each entry is the proximity
     * of that cell's 5th-percentile depth, i.e. how near its nearest content
     * is: 0 = far, 1 = nearest in frame. */
    float zones[9] = {0};

    /* Geometry the numbers were computed over. */
    int src_w = 0;
    int src_h = 0;
    int roi_x = 0;
    int roi_y = 0;
    int roi_w = 0;
    int roi_h = 0;
};

/*
 * Reduce a depth map to the statistics above.
 *
 * Range stabilisation uses p02/p98 rather than min/max: a single hot pixel at
 * either end would otherwise rescale the whole frame and make proximity jump
 * between consecutive frames that look identical.
 *
 *   proximity = clamp((p98 - d) / (p98 - p02), 0, 1)
 *
 * `near_threshold` is the proximity at which a pixel counts as near;
 * `near_ratio_threshold` is the fraction of such pixels that makes
 * near_present true.
 */
DepthStats computeStats(const std::vector<float>& depth, int width, int height,
                        const Roi& roi, int src_w, int src_h,
                        float near_threshold, float near_ratio_threshold);

/* The "depth":{...} object, without the surrounding braces of the envelope. */
std::string buildDepthObject(const DepthStats& stats);

/* Full MQTT results document. */
std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            float inference_time_ms, const DepthStats& stats);

}  // namespace depth

#endif  // _DEPTH_PAYLOAD_H_
