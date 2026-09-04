#include "depth_payload.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace depth {

namespace {

/* Percentile of a scratch buffer that may be reordered. q in [0,1]. */
float percentile(std::vector<float>& scratch, float q) {
    if (scratch.empty()) return 0.0f;
    size_t k = static_cast<size_t>(q * static_cast<float>(scratch.size() - 1) + 0.5f);
    if (k >= scratch.size()) k = scratch.size() - 1;
    std::nth_element(scratch.begin(), scratch.begin() + k, scratch.end());
    return scratch[k];
}

/* proximity = clamp((p98 - d) / (p98 - p02), 0, 1), with span = p98 - p02.
 * A flat frame (span 0) has no near and no far, so everything reads 0. */
inline float proximity_of(float d, float p02, float span) {
    if (span <= 0.0f) return 0.0f;
    float v = (p02 + span - d) / span;
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return v;
}

}  // namespace

DepthStats computeStats(const std::vector<float>& depth, int width, int height,
                        const Roi& roi, int src_w, int src_h,
                        float near_threshold, float near_ratio_threshold) {
    DepthStats st;
    st.src_w = src_w;
    st.src_h = src_h;
    st.roi_x = roi.x;
    st.roi_y = roi.y;
    st.roi_w = roi.w;
    st.roi_h = roi.h;

    const size_t n = depth.size();
    if (n == 0 || width <= 0 || height <= 0 ||
        static_cast<size_t>(width) * height != n) {
        return st;
    }

    double sum   = 0.0;
    float  vmin  = depth[0];
    float  vmax  = depth[0];
    for (size_t i = 0; i < n; i++) {
        const float d = depth[i];
        sum += d;
        if (d < vmin) vmin = d;
        if (d > vmax) vmax = d;
    }
    st.min  = vmin;
    st.max  = vmax;
    st.mean = static_cast<float>(sum / static_cast<double>(n));

    std::vector<float> scratch(depth);
    st.p02 = percentile(scratch, 0.02f);
    st.p50 = percentile(scratch, 0.50f);
    st.p98 = percentile(scratch, 0.98f);

    const float span = st.p98 - st.p02;

    size_t near_count = 0;
    if (span > 0.0f) {
        for (size_t i = 0; i < n; i++) {
            if (proximity_of(depth[i], st.p02, span) >= near_threshold) near_count++;
        }
    }
    st.near_ratio   = static_cast<float>(near_count) / static_cast<float>(n);
    st.near_present = st.near_ratio >= near_ratio_threshold;

    /* 3x3 grid. Each cell reports the proximity of its 5th-percentile depth --
     * its nearest content, with the same p02/p98 stabilisation applied to the
     * whole frame so the nine numbers are comparable with each other. */
    std::vector<float> cell;
    cell.reserve((static_cast<size_t>(width) / 3 + 1) * (static_cast<size_t>(height) / 3 + 1));
    for (int gy = 0; gy < 3; gy++) {
        const int y0 = height * gy / 3;
        const int y1 = height * (gy + 1) / 3;
        for (int gx = 0; gx < 3; gx++) {
            const int x0 = width * gx / 3;
            const int x1 = width * (gx + 1) / 3;
            cell.clear();
            for (int y = y0; y < y1; y++) {
                const float* row = depth.data() + static_cast<size_t>(y) * width;
                for (int x = x0; x < x1; x++) cell.push_back(row[x]);
            }
            const float d05 = percentile(cell, 0.05f);
            st.zones[gy * 3 + gx] = proximity_of(d05, st.p02, span);
        }
    }

    return st;
}

std::string buildDepthObject(const DepthStats& stats) {
    std::ostringstream j;
    j << std::fixed;
    j << "{";
    j << "\"unit\":\"relative\",";
    j << "\"smaller_is_nearer\":true,";
    j << "\"source_size\":[" << stats.src_w << "," << stats.src_h << "],";
    j << "\"valid_roi\":[" << stats.roi_x << "," << stats.roi_y << ","
      << stats.roi_w << "," << stats.roi_h << "],";
    j << std::setprecision(4);
    j << "\"min\":" << stats.min << ",";
    j << "\"max\":" << stats.max << ",";
    j << "\"mean\":" << stats.mean << ",";
    j << "\"p02\":" << stats.p02 << ",";
    j << "\"p50\":" << stats.p50 << ",";
    j << "\"p98\":" << stats.p98 << ",";
    j << std::setprecision(4) << "\"near_ratio\":" << stats.near_ratio << ",";
    j << "\"near_present\":" << (stats.near_present ? "true" : "false") << ",";
    j << "\"zones\":[";
    for (int i = 0; i < 9; i++) {
        if (i) j << ",";
        j << stats.zones[i];
    }
    j << "]";
    j << "}";
    return j.str();
}

std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            float inference_time_ms, const DepthStats& stats) {
    std::ostringstream j;
    j << std::fixed;
    j << "{";
    j << "\"timestamp\":" << timestamp_ms << ",";
    j << "\"frame_id\":" << frame_id << ",";
    j << std::setprecision(1) << "\"inference_time_ms\":" << inference_time_ms << ",";
    j << "\"depth\":" << buildDepthObject(stats);
    j << "}";
    return j.str();
}

}  // namespace depth
