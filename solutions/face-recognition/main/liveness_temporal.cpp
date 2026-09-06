#include "liveness_temporal.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace face_recognition {

namespace {

float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

// Clipped linear ramp; returns 0 below `lo`, 1 above `hi`, and is safe when the
// caller configures a degenerate lo >= hi (then it degrades to a step at `hi`).
float ramp(float v, float lo, float hi) {
    if (!(hi > lo)) return v >= hi ? 1.f : 0.f;
    return clamp01((v - lo) / (hi - lo));
}

}  // namespace

void updateTexture(LivenessState& state, float ensemble_mean, const LivenessConfig& cfg) {
    const float a = clamp01(cfg.texture_ema_alpha);
    if (!state.has_texture) {
        state.texture_ema = ensemble_mean;
        state.has_texture = true;
    } else {
        state.texture_ema = a * ensemble_mean + (1.f - a) * state.texture_ema;
    }
    state.texture_samples++;
}

float normalizedMotion(float residual, float correlation, const LivenessConfig& cfg) {
    // Magnitude gate: how far the post-similarity residual sits above the
    // detector's own jitter floor.
    const float magnitude = ramp(residual, cfg.motion_noise_floor, cfg.motion_high);
    // Structure gate: independent per-point detector noise decorrelates across
    // points, genuine facial motion does not. Both gates must open, so pure
    // jitter (large residual, near-zero correlation) scores ~0.
    const float structure = ramp(correlation, cfg.correlation_low, cfg.correlation_high);
    return clamp01(magnitude * structure);
}

MotionFeatures updateMotion(LivenessState& state, uint64_t ts_ms, const Landmarks5& kps,
                            float face_px, const LivenessConfig& cfg) {
    KeypointSample s;
    s.ts_ms = ts_ms;
    s.face_px = face_px > 1.f ? face_px : 1.f;
    s.kps = kps;

    // Monotonic guard: a timestamp that went backwards means the caller's clock
    // was reset, and mixing the two eras would fabricate a huge residual.
    if (!state.keypoints.empty() && ts_ms < state.keypoints.back().ts_ms) {
        state.keypoints.clear();
    }
    state.keypoints.push_back(s);

    const uint64_t window_ms =
        static_cast<uint64_t>(std::max(0.f, cfg.motion_window_sec) * 1000.f);
    while (!state.keypoints.empty() && state.keypoints.front().ts_ms + window_ms < ts_ms) {
        state.keypoints.pop_front();
    }

    MotionFeatures out;
    const size_t n = state.keypoints.size();
    if (static_cast<int>(n) < kMinMotionSamples) {
        state.motion = out;  // valid = false
        return out;
    }

    // residual[pair][point] = displacement left over after the best similarity
    // transform between adjacent observations, normalized by face short side.
    const size_t pairs = n - 1;
    std::vector<std::array<float, 5>> rx(pairs), ry(pairs);

    double sum_sq = 0.0;
    size_t counted = 0;
    for (size_t p = 0; p + 1 < n; ++p) {
        const KeypointSample& a = state.keypoints[p];
        const KeypointSample& b = state.keypoints[p + 1];

        // Centroids.
        float ax = 0.f, ay = 0.f, bx = 0.f, by = 0.f;
        for (int i = 0; i < 5; ++i) {
            ax += a.kps[i].x; ay += a.kps[i].y;
            bx += b.kps[i].x; by += b.kps[i].y;
        }
        ax /= 5.f; ay /= 5.f; bx /= 5.f; by /= 5.f;

        // Closed-form least-squares similarity (Umeyama in 2D): with centred
        // point sets, s*cos and s*sin fall straight out of the dot and cross
        // sums. Translation is absorbed by the centring, so it never appears in
        // the residual -- a face that walks across the frame is not "motion".
        double dot = 0.0, cross = 0.0, den = 0.0;
        for (int i = 0; i < 5; ++i) {
            const double axi = a.kps[i].x - ax, ayi = a.kps[i].y - ay;
            const double bxi = b.kps[i].x - bx, byi = b.kps[i].y - by;
            dot += axi * bxi + ayi * byi;
            cross += axi * byi - ayi * bxi;
            den += axi * axi + ayi * ayi;
        }
        double c = 1.0, s2 = 0.0;
        if (den > 1e-9) {
            c = dot / den;
            s2 = cross / den;
        }

        const float norm = b.face_px;
        for (int i = 0; i < 5; ++i) {
            const double axi = a.kps[i].x - ax, ayi = a.kps[i].y - ay;
            const double bxi = b.kps[i].x - bx, byi = b.kps[i].y - by;
            const double px = c * axi - s2 * ayi;
            const double py = s2 * axi + c * ayi;
            const float ex = static_cast<float>((bxi - px) / norm);
            const float ey = static_cast<float>((byi - py) / norm);
            rx[p][i] = ex;
            ry[p][i] = ey;
            sum_sq += static_cast<double>(ex) * ex + static_cast<double>(ey) * ey;
            ++counted;
        }
    }

    out.residual = counted ? static_cast<float>(std::sqrt(sum_sq / (2.0 * counted))) : 0.f;

    // Mean off-diagonal entry of the TEMPORAL correlation matrix: each adjacent
    // pair contributes one 10-component residual vector (five points x, then
    // five points y), and C[p][q] is the Pearson correlation between the vectors
    // of pair p and pair q. The mean is taken over p != q.
    //
    // Why between time steps and not between points: after the similarity fit
    // the per-point residuals are constrained to sum to zero, which pins the
    // mean pairwise POINT correlation at -1/(n-1) = -0.25 for five points no
    // matter what the face is doing. That statistic carries no signal. The
    // temporal one does:
    //   detector jitter   -> successive residual vectors are independent, ~0
    //   facial deformation-> successive residual vectors share a direction, ~+1
    double corr_sum = 0.0;
    int corr_n = 0;
    std::vector<std::vector<double>> vecs(pairs, std::vector<double>(10, 0.0));
    for (size_t p = 0; p < pairs; ++p) {
        for (int i = 0; i < 5; ++i) {
            vecs[p][i] = rx[p][i];
            vecs[p][5 + i] = ry[p][i];
        }
    }
    std::vector<double> mean(pairs, 0.0), sd(pairs, 0.0);
    for (size_t p = 0; p < pairs; ++p) {
        double m = 0.0;
        for (double v : vecs[p]) m += v;
        m /= 10.0;
        double var = 0.0;
        for (double v : vecs[p]) var += (v - m) * (v - m);
        var /= 10.0;
        mean[p] = m;
        sd[p] = std::sqrt(var);
    }
    for (size_t p = 0; p < pairs; ++p) {
        for (size_t q = p + 1; q < pairs; ++q) {
            if (sd[p] < 1e-12 || sd[q] < 1e-12) continue;
            double cov = 0.0;
            for (int k = 0; k < 10; ++k) cov += (vecs[p][k] - mean[p]) * (vecs[q][k] - mean[q]);
            cov /= 10.0;
            corr_sum += cov / (sd[p] * sd[q]);
            ++corr_n;
        }
    }
    // A negative mean correlation is anti-coherent motion, not structure; the
    // score ramp clips it to zero anyway, but reporting it unclipped keeps the
    // captured feature honest for offline fitting.
    out.correlation = corr_n ? static_cast<float>(corr_sum / corr_n) : 0.f;
    out.score = normalizedMotion(out.residual, out.correlation, cfg);
    out.valid = true;

    state.motion = out;
    return out;
}

bool updateBlink(LivenessState& state, bool have_ear, float ear, const LivenessConfig& cfg) {
    state.blink_this_frame = false;
    if (!have_ear) {
        // A skipped or failed FaceMesh sample is not evidence of an open eye.
        // Advancing the counter here would turn dropped frames into blinks.
        return false;
    }
    state.has_ear = true;
    state.ear = ear;

    if (ear < cfg.ear_threshold) {
        state.closed_samples++;
        if (state.closed_samples > cfg.blink_max_samples) {
            // Sustained closure: drowsiness, occlusion or a photo of closed
            // eyes. Not a blink, and it must not become one when the eyes
            // eventually reopen.
            state.closure_overlong = true;
        }
        return false;
    }

    const int closed = state.closed_samples;
    const bool overlong = state.closure_overlong;
    state.closed_samples = 0;
    state.closure_overlong = false;

    if (!overlong && closed >= cfg.blink_min_samples && closed <= cfg.blink_max_samples) {
        state.blink_seen = true;   // latched for the lifetime of the track
        state.blink_this_frame = true;
        return true;
    }
    return false;
}

LivenessResult fuseLiveness(LivenessState& state, uint64_t now_ms, const LivenessConfig& cfg) {
    LivenessResult r;
    r.blink = state.blink_seen;
    if (state.has_texture) {
        r.has_texture = true;
        r.texture = state.texture_ema;
    }
    if (state.motion.valid) {
        r.has_motion = true;
        r.motion = state.motion.score;
    }

    if (!state.started) {
        state.started = true;
        state.first_seen_ms = now_ms;
    }
    const uint64_t elapsed_ms = now_ms >= state.first_seen_ms ? now_ms - state.first_seen_ms : 0;
    const bool timed_out =
        elapsed_ms >= static_cast<uint64_t>(std::max(0.f, cfg.timeout_sec) * 1000.f);

    // Gate 1: too few texture samples to say anything at all. Note this is a
    // gate on the verdict only -- motion and blink keep accumulating.
    if (state.texture_samples < cfg.min_samples) {
        state.decision = "pending";
        state.has_score = false;
        state.reason = "insufficient_samples";
        r.decision = state.decision;
        r.has_score = false;
        r.reason = state.reason;
        return r;
    }

    // Motion counts only once its residual clears the detector noise floor.
    const bool motion_usable =
        state.motion.valid && state.motion.residual > cfg.motion_noise_floor;

    const float w_t = std::max(0.f, cfg.w_texture);
    const float w_m = motion_usable ? std::max(0.f, cfg.w_motion) : 0.f;
    const float w_d = (cfg.depth_enabled && state.has_depth) ? std::max(0.f, cfg.w_depth) : 0.f;
    float wsum = w_t + w_m + w_d;
    if (!(wsum > 0.f)) wsum = 1.f;

    float base = (w_t * state.texture_ema + w_m * state.motion.score +
                  w_d * state.depth_score) / wsum;
    // A blink is the strongest single piece of passive evidence available, so it
    // is added as a bonus rather than as a weighted term: with the default bonus
    // of 1.0 the score saturates and the verdict cannot be dragged back below
    // T_spoof by a middling texture reading.
    if (state.blink_seen) base += std::max(0.f, cfg.blink_bonus);
    const float score = clamp01(base);

    r.has_score = true;
    r.score = score;

    if (state.blink_seen) {
        state.decision = "live";
        state.reason = "blink";
    } else if (!motion_usable && !timed_out) {
        // Still inside the observation window with no usable motion: withhold a
        // verdict rather than judging on texture alone this early.
        state.decision = "pending";
        state.has_score = true;
        state.score = score;
        state.reason = "awaiting_motion";
        r.decision = state.decision;
        r.reason = state.reason;
        return r;
    } else {
        // Hysteresis. Absence of motion at timeout is explicitly NOT a penalty:
        // `base` above already renormalized the weights over the evidence that
        // actually exists, so a still subject is judged on texture (+ depth).
        const std::string prev = state.decision;
        if (prev == "live") {
            state.decision = score < cfg.t_spoof ? "spoof" : "live";
        } else if (prev == "spoof") {
            state.decision = score > cfg.t_live ? "live" : "spoof";
        } else {
            if (score >= cfg.t_live) {
                state.decision = "live";
            } else if (score <= cfg.t_spoof) {
                state.decision = "spoof";
            } else {
                state.decision = "pending";
            }
        }
        if (state.decision == "pending") {
            state.reason = "hysteresis_band";
        } else if (!motion_usable) {
            state.reason = "texture_timeout";
        } else {
            state.reason = "fused";
        }
    }

    state.has_score = true;
    state.score = score;
    r.decision = state.decision;
    r.reason = state.reason;
    return r;
}

}  // namespace face_recognition
