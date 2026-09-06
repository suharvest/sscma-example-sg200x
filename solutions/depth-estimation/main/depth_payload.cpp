#include "depth_payload.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace depth {

/* Histogram resolution. kBins covers the whole frame (percentiles and the
 * near-count cut); kZoneBins is per 3x3 cell. Both are 2048: a coarser zone
 * histogram quantises the nine numbers visibly, because proximity() divides
 * the bin width by the frame's span and so amplifies it -- at 512 bins,
 * neighbouring cells on a low-contrast frame collapsed onto one value. Cell
 * counts (~n/9) stay far below UINT16_MAX. */
static constexpr size_t kBins     = 2048;
static constexpr size_t kZoneBins = 2048;

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

    /* Everything below comes out of one histogram pass instead of copying the
     * map and running nth_element twelve times (three frame percentiles, nine
     * cell percentiles) plus a separate near-count pass with a divide per
     * pixel. Bin width is (max-min)/kBins, so a percentile is accurate to
     * about 0.05% of the frame's depth range -- far below anything these
     * relative-depth numbers are read to. */
    static std::vector<uint32_t> hist(kBins);
    static std::vector<uint16_t> zhist(9 * kZoneBins);
    std::fill(hist.begin(), hist.end(), 0u);
    std::fill(zhist.begin(), zhist.end(), uint16_t{0});

    const float range = st.max - st.min;
    if (!(range > 0.0f)) {
        /* Flat map: every percentile is the single value, nothing is "near". */
        st.p02 = st.p50 = st.p98 = st.min;
        st.near_ratio   = 0.0f;
        st.near_present = false;
        for (int i = 0; i < 9; i++) st.zones[i] = 0.0f;
        return st;
    }

    const float bin_scale  = static_cast<float>(kBins - 1) / range;
    const float zbin_scale = static_cast<float>(kZoneBins - 1) / range;

    /* Row-major walk so the zone index is derived from the loop counters
     * rather than a per-pixel divide. The cut points must be the same integer
     * divisions the cell loop used before (height*g/3), not the algebraically
     * "equivalent" y*3 >= height: for 224 rows those differ by one row, which
     * moves ~150 pixels between cells and shifts every zone value far more
     * than the histogram's own quantisation does. */
    const int ycut1 = height / 3, ycut2 = height * 2 / 3;
    const int xcut1 = width / 3,  xcut2 = width * 2 / 3;
    for (int y = 0; y < height; y++) {
        const int gy = (y >= ycut2) ? 2 : (y >= ycut1 ? 1 : 0);
        const float* row = depth.data() + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; x++) {
            const float d = row[x] - st.min;
            hist[static_cast<size_t>(d * bin_scale)]++;
            const int gx = (x >= xcut2) ? 2 : (x >= xcut1 ? 1 : 0);
            uint16_t& c = zhist[static_cast<size_t>(gy * 3 + gx) * kZoneBins +
                                static_cast<size_t>(d * zbin_scale)];
            if (c != UINT16_MAX) c++;  // saturate; a cell never fills a bin
        }
    }

    const auto bin_value = [&](size_t bin, size_t bins) {
        return st.min + (static_cast<float>(bin) + 0.5f) * range /
                            static_cast<float>(bins - 1);
    };
    /* Smallest bin whose cumulative count reaches q of the total. */
    const auto hist_percentile = [&](const uint32_t* h, size_t bins,
                                     uint64_t total, float q) {
        const uint64_t want = static_cast<uint64_t>(q * static_cast<float>(total));
        uint64_t acc = 0;
        for (size_t b = 0; b < bins; b++) {
            acc += h[b];
            if (acc > want) return bin_value(b, bins);
        }
        return bin_value(bins - 1, bins);
    };

    st.p02 = hist_percentile(hist.data(), kBins, n, 0.02f);
    st.p50 = hist_percentile(hist.data(), kBins, n, 0.50f);
    st.p98 = hist_percentile(hist.data(), kBins, n, 0.98f);

    const float span = st.p98 - st.p02;

    /* proximity(d) >= t  <=>  d <= p98 - t*span, so the near count is a
     * prefix sum of the same histogram -- no second pass, no per-pixel divide. */
    size_t near_count = 0;
    if (span > 0.0f) {
        const float cut = st.p98 - near_threshold * span;
        if (cut >= st.min) {
            const float fb = (cut - st.min) * bin_scale;
            const size_t last =
                (fb >= static_cast<float>(kBins - 1)) ? kBins - 1
                                                      : static_cast<size_t>(fb);
            for (size_t b = 0; b <= last; b++) near_count += hist[b];
        }
    }
    st.near_ratio   = static_cast<float>(near_count) / static_cast<float>(n);
    st.near_present = st.near_ratio >= near_ratio_threshold;

    /* 3x3 grid. Each cell reports the proximity of its 5th-percentile depth --
     * its nearest content, with the same p02/p98 stabilisation applied to the
     * whole frame so the nine numbers are comparable with each other. */
    const float inv_span = (span > 0.0f) ? 1.0f / span : 0.0f;
    for (int g = 0; g < 9; g++) {
        const uint16_t* zh = zhist.data() + static_cast<size_t>(g) * kZoneBins;
        uint64_t cnt = 0;
        for (size_t b = 0; b < kZoneBins; b++) cnt += zh[b];
        if (cnt == 0) {
            st.zones[g] = 0.0f;
            continue;
        }
        const uint64_t want = static_cast<uint64_t>(0.05f * static_cast<float>(cnt));
        uint64_t acc  = 0;
        float    d05  = st.min;
        for (size_t b = 0; b < kZoneBins; b++) {
            acc += zh[b];
            if (acc > want) {
                d05 = st.min + (static_cast<float>(b) + 0.5f) * range /
                                   static_cast<float>(kZoneBins - 1);
                break;
            }
        }
        float v = (st.p02 + span - d05) * inv_span;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        st.zones[g] = v;
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
