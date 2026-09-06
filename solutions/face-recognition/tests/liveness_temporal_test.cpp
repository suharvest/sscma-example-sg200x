// Host-only tests for the liveness v2 temporal core and the calibration writer.
//
// No SSCMA, no CVI, no OpenCV, no camera. Build and run with:
//   cmake -S tests -B tests/build && cmake --build tests/build && ctest --test-dir tests/build

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "liveness_capture_writer.h"
#include "liveness_temporal.h"

using namespace face_recognition;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #cond   \
                      << std::endl;                                            \
        }                                                                      \
    } while (0)

#define CHECK_EQ_STR(a, b)                                                     \
    do {                                                                       \
        ++g_checks;                                                            \
        const std::string _a = (a);                                            \
        const std::string _b = (b);                                            \
        if (_a != _b) {                                                        \
            ++g_failures;                                                      \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  got '"    \
                      << _a << "' want '" << _b << "'" << std::endl;           \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
    do {                                                                       \
        ++g_checks;                                                            \
        const double _a = (a), _b = (b);                                       \
        if (!(std::fabs(_a - _b) <= (eps))) {                                  \
            ++g_failures;                                                      \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  got "     \
                      << _a << " want " << _b << " +/- " << (eps) << std::endl; \
        }                                                                      \
    } while (0)

static void run(const char* name, void (*fn)()) {
    const int before = g_failures;
    fn();
    std::cout << (g_failures == before ? "ok   " : "FAIL ") << name << std::endl;
}

// A neutral five-point face: eyes, nose, mouth corners, roughly to scale for a
// 100 px face.
static Landmarks5 baseFace() {
    Landmarks5 k;
    k[0] = {130.f, 130.f};  // left eye
    k[1] = {170.f, 130.f};  // right eye
    k[2] = {150.f, 155.f};  // nose
    k[3] = {135.f, 175.f};  // left mouth
    k[4] = {165.f, 175.f};  // right mouth
    return k;
}

static Landmarks5 transform(const Landmarks5& in, float scale, float deg, float tx, float ty) {
    const float rad = deg * 3.14159265358979f / 180.f;
    const float c = std::cos(rad) * scale, s = std::sin(rad) * scale;
    Landmarks5 out;
    for (int i = 0; i < 5; ++i) {
        const float x = in[i].x - 150.f, y = in[i].y - 150.f;
        out[i] = {150.f + c * x - s * y + tx, 150.f + s * x + c * y + ty};
    }
    return out;
}

// ---------------------------------------------------------------------------
// Motion
// ---------------------------------------------------------------------------

// A rigid translate/rotate/scale sequence is exactly what the similarity fit
// removes. A face merely walking toward the camera must not read as "motion".
static void test_motion_rigid_is_removed() {
    LivenessConfig cfg;
    LivenessState st;
    const Landmarks5 base = baseFace();
    MotionFeatures m;
    for (int i = 0; i < 12; ++i) {
        const Landmarks5 k = transform(base, 1.f + 0.02f * i, 3.f * i, 5.f * i, -2.f * i);
        m = updateMotion(st, 100ull * i, k, 100.f, cfg);
    }
    CHECK(m.valid);
    CHECK(m.residual < 1e-4f);
    CHECK_NEAR(m.score, 0.0, 1e-6);
}

// Fewer than kMinMotionSamples observations yields no verdict at all.
static void test_motion_needs_window() {
    LivenessConfig cfg;
    LivenessState st;
    const Landmarks5 base = baseFace();
    for (int i = 0; i < kMinMotionSamples - 1; ++i) {
        const MotionFeatures m = updateMotion(st, 100ull * i, base, 100.f, cfg);
        CHECK(!m.valid);
    }
    const MotionFeatures m = updateMotion(st, 100ull * (kMinMotionSamples - 1), base, 100.f, cfg);
    CHECK(m.valid);
}

// Independent per-point jitter produces residual but near-zero correlation, and
// the structure gate must keep the score at the floor. This is pitfall 1: a
// residual-only threshold would call detector noise "motion".
static void test_motion_uncorrelated_jitter_scores_low() {
    LivenessConfig cfg;
    LivenessState st;
    const Landmarks5 base = baseFace();
    unsigned seed = 12345u;
    auto rnd = [&]() {
        seed = seed * 1103515245u + 12345u;
        return static_cast<float>((seed >> 16) % 2000) / 1000.f - 1.f;  // -1..1
    };
    MotionFeatures m;
    for (int i = 0; i < 16; ++i) {
        Landmarks5 k = base;
        for (int p = 0; p < 5; ++p) {
            k[p].x += rnd() * 0.8f;
            k[p].y += rnd() * 0.8f;
        }
        m = updateMotion(st, 100ull * i, k, 100.f, cfg);
    }
    CHECK(m.valid);
    CHECK(m.residual > cfg.motion_noise_floor);   // there IS displacement
    CHECK(m.correlation < cfg.correlation_high);  // but it is not structured
    CHECK(m.score < 0.5f);
}

// A localized deformation of the mouth points whose direction PERSISTS across
// frames is structured: consecutive residual vectors agree, correlation goes to
// ~1 and the score saturates. The amplitude here is exaggerated so the residual
// also clears the (uncalibrated) motion_high default in one test.
static void test_motion_structured_deformation_scores_high() {
    LivenessConfig cfg;
    LivenessState st;
    const Landmarks5 base = baseFace();
    MotionFeatures m;
    for (int i = 0; i < 16; ++i) {
        Landmarks5 k = base;
        k[3].y += 20.0f * i;
        k[4].y += 20.0f * i;
        m = updateMotion(st, 100ull * i, k, 100.f, cfg);
    }
    CHECK(m.valid);
    CHECK(m.residual > cfg.motion_high);
    CHECK(m.correlation > cfg.correlation_high);
    CHECK(m.score > 0.9f);
}

// Samples older than the window are evicted, and a backwards timestamp (clock
// reset) discards the window instead of fabricating a huge residual.
static void test_motion_window_eviction_and_time_gap() {
    LivenessConfig cfg;
    cfg.motion_window_sec = 1.0f;
    LivenessState st;
    const Landmarks5 base = baseFace();
    for (int i = 0; i < 20; ++i) updateMotion(st, 100ull * i, base, 100.f, cfg);
    CHECK(st.keypoints.size() <= 11);   // 1.0 s at 10 Hz, inclusive of both ends

    updateMotion(st, 10ull, base, 100.f, cfg);  // clock went backwards
    CHECK(st.keypoints.size() == 1);
}

// ---------------------------------------------------------------------------
// Blink
// ---------------------------------------------------------------------------

static void feedEar(LivenessState& st, const LivenessConfig& cfg, const std::vector<float>& ears,
                    std::vector<bool>* blinks = nullptr) {
    for (float e : ears) {
        const bool b = updateBlink(st, true, e, cfg);
        if (blinks) blinks->push_back(b);
    }
}

// open -> 1..3 closed -> open is a blink; the latch survives.
static void test_blink_short_closures() {
    LivenessConfig cfg;
    for (int n = 1; n <= 3; ++n) {
        LivenessState st;
        std::vector<float> ears{0.30f};
        for (int i = 0; i < n; ++i) ears.push_back(0.10f);
        ears.push_back(0.30f);
        std::vector<bool> blinks;
        feedEar(st, cfg, ears, &blinks);
        CHECK(st.blink_seen);
        CHECK(blinks.back());
    }
}

// A closure longer than blink_max_samples is not a blink, and reopening must
// not retroactively make it one.
static void test_blink_overlong_closure_is_not_a_blink() {
    LivenessConfig cfg;
    LivenessState st;
    feedEar(st, cfg, {0.30f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.30f});
    CHECK(!st.blink_seen);
    CHECK(st.closed_samples == 0);
    // A proper blink afterwards still registers.
    feedEar(st, cfg, {0.10f, 0.30f});
    CHECK(st.blink_seen);
}

// A skipped FaceMesh sample is not an open eye: it must not close out a dip.
static void test_blink_missing_samples_do_not_advance() {
    LivenessConfig cfg;
    LivenessState st;
    CHECK(!updateBlink(st, true, 0.10f, cfg));
    CHECK(st.closed_samples == 1);
    CHECK(!updateBlink(st, false, 0.f, cfg));   // FaceMesh skipped this frame
    CHECK(st.closed_samples == 1);
    CHECK(updateBlink(st, true, 0.30f, cfg));
    CHECK(st.blink_seen);
}

// ---------------------------------------------------------------------------
// Fusion
// ---------------------------------------------------------------------------

// Production arms the track on its first visible frame (main.cpp's
// sample_liveness), so first_seen_ms is set before any fusion happens. Tests
// that reason about the timeout have to do the same or elapsed is always 0.
static void arm(LivenessState& st) {
    st.started = true;
    st.first_seen_ms = 0;
}

static void test_fuse_pending_until_three_samples() {
    LivenessConfig cfg;
    LivenessState st;
    arm(st);
    for (int i = 0; i < 2; ++i) {
        updateTexture(st, 0.95f, cfg);
        const LivenessResult r = fuseLiveness(st, 100ull * i, cfg);
        CHECK_EQ_STR(r.decision, "pending");
        CHECK_EQ_STR(r.reason, "insufficient_samples");
        CHECK(!r.has_score);
    }
    updateTexture(st, 0.95f, cfg);
    const LivenessResult r = fuseLiveness(st, 5000ull, cfg);
    CHECK(r.decision != "pending");
    CHECK(r.has_score);
}

static void test_fuse_ema() {
    LivenessConfig cfg;
    cfg.texture_ema_alpha = 0.5f;
    LivenessState st;
    updateTexture(st, 1.0f, cfg);
    CHECK_NEAR(st.texture_ema, 1.0, 1e-6);   // first sample seeds, not blends
    updateTexture(st, 0.0f, cfg);
    CHECK_NEAR(st.texture_ema, 0.5, 1e-6);
    updateTexture(st, 0.0f, cfg);
    CHECK_NEAR(st.texture_ema, 0.25, 1e-6);
    CHECK(st.texture_samples == 3);
}

// A blink is decisive even when texture is poor.
static void test_fuse_blink_overrides_weak_texture() {
    LivenessConfig cfg;
    LivenessState st;
    for (int i = 0; i < 3; ++i) updateTexture(st, 0.20f, cfg);
    st.blink_seen = true;
    const LivenessResult r = fuseLiveness(st, 1000ull, cfg);
    CHECK_EQ_STR(r.decision, "live");
    CHECK_EQ_STR(r.reason, "blink");
    CHECK(r.blink);
    CHECK_NEAR(r.score, 1.0, 1e-6);
}

// The absence of a blink NEVER produces a spoof on its own: good texture with no
// blink and no motion resolves live at the timeout.
static void test_fuse_no_blink_is_not_a_spoof() {
    LivenessConfig cfg;
    LivenessState st;
    arm(st);
    for (int i = 0; i < 3; ++i) updateTexture(st, 0.90f, cfg);
    // Before the timeout, with no usable motion, the verdict is withheld.
    LivenessResult r = fuseLiveness(st, 500ull, cfg);
    CHECK_EQ_STR(r.decision, "pending");
    CHECK_EQ_STR(r.reason, "awaiting_motion");
    // At the timeout it is judged on texture alone, with no motion penalty.
    r = fuseLiveness(st, 2500ull, cfg);
    CHECK_EQ_STR(r.decision, "live");
    CHECK_EQ_STR(r.reason, "texture_timeout");
    CHECK_NEAR(r.score, 0.90, 1e-5);   // renormalized over texture only
}

// Bad texture at the timeout does reach spoof.
static void test_fuse_texture_timeout_spoof() {
    LivenessConfig cfg;
    LivenessState st;
    arm(st);
    for (int i = 0; i < 3; ++i) updateTexture(st, 0.10f, cfg);
    const LivenessResult r = fuseLiveness(st, 3000ull, cfg);
    CHECK_EQ_STR(r.decision, "spoof");
    CHECK_EQ_STR(r.reason, "texture_timeout");
}

// T_live / T_spoof hysteresis: once live, a score in the band stays live; only a
// drop below T_spoof flips it, and only a rise above T_live flips it back.
static void test_fuse_hysteresis() {
    LivenessConfig cfg;
    LivenessState st;
    arm(st);
    for (int i = 0; i < 3; ++i) updateTexture(st, 0.90f, cfg);
    LivenessResult r = fuseLiveness(st, 3000ull, cfg);
    CHECK_EQ_STR(r.decision, "live");

    // Drift into the band (0.45 .. 0.65): stays live.
    st.texture_ema = 0.55f;
    r = fuseLiveness(st, 3100ull, cfg);
    CHECK_EQ_STR(r.decision, "live");

    // Below T_spoof: flips.
    st.texture_ema = 0.30f;
    r = fuseLiveness(st, 3200ull, cfg);
    CHECK_EQ_STR(r.decision, "spoof");

    // Back into the band: stays spoof.
    st.texture_ema = 0.55f;
    r = fuseLiveness(st, 3300ull, cfg);
    CHECK_EQ_STR(r.decision, "spoof");

    // Above T_live: flips back.
    st.texture_ema = 0.80f;
    r = fuseLiveness(st, 3400ull, cfg);
    CHECK_EQ_STR(r.decision, "live");
}

// Motion below the noise floor is ignored rather than dragging the score down.
static void test_fuse_motion_below_floor_is_ignored() {
    LivenessConfig cfg;
    LivenessState st;
    arm(st);
    for (int i = 0; i < 3; ++i) updateTexture(st, 0.90f, cfg);
    st.motion.valid = true;
    st.motion.residual = cfg.motion_noise_floor * 0.5f;
    st.motion.score = 0.f;
    const LivenessResult r = fuseLiveness(st, 3000ull, cfg);
    CHECK_EQ_STR(r.decision, "live");
    CHECK_NEAR(r.score, 0.90, 1e-5);   // motion contributed no weight at all
}

// Usable motion participates with its configured weight.
static void test_fuse_motion_weighted() {
    LivenessConfig cfg;
    LivenessState st;
    for (int i = 0; i < 3; ++i) updateTexture(st, 0.80f, cfg);
    st.motion.valid = true;
    st.motion.residual = cfg.motion_high;
    st.motion.score = 1.0f;
    const LivenessResult r = fuseLiveness(st, 500ull, cfg);
    CHECK_EQ_STR(r.decision, "live");
    CHECK_EQ_STR(r.reason, "fused");
    CHECK_NEAR(r.score, 0.70 * 0.80 + 0.30 * 1.0, 1e-5);
    CHECK(r.has_motion);
}

// A truncated track is simply a fresh state: nothing leaks across.
static void test_track_retirement_is_a_fresh_state() {
    LivenessConfig cfg;
    LivenessState st;
    for (int i = 0; i < 3; ++i) updateTexture(st, 0.95f, cfg);
    st.blink_seen = true;
    (void)fuseLiveness(st, 3000ull, cfg);
    CHECK_EQ_STR(st.decision, "live");

    st = LivenessState{};   // what erasing the map entry does
    const LivenessResult r = fuseLiveness(st, 4000ull, cfg);
    CHECK_EQ_STR(r.decision, "pending");
    CHECK(!r.blink);
    CHECK(!r.has_texture);
}

// ---------------------------------------------------------------------------
// Capture writer
// ---------------------------------------------------------------------------

static void test_capture_label_and_bounds() {
    LivenessCaptureWriter w;
    std::string err;
    const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                             "/liveness_capture_test.jsonl";
    std::remove(path.c_str());

    CHECK(!w.start(path, "hologram", 5, 60, 0, &err));   // bad label
    CHECK(!w.active());
    CHECK(w.start(path, "real", 999, 60, 0, &err));      // clamped to the maximum
    CHECK(w.seconds() == 60);
    CHECK(!w.start(path, "real", 5, 60, 0, &err));       // already running
    w.stop();

    CHECK(w.start(path, "print", 0, 60, 0, &err));       // clamped up to 1
    CHECK(w.seconds() == 1);
    w.stop();
}

static void test_capture_row_nulls_and_append() {
    CaptureRow row;
    row.has_v2 = true;
    row.p_tex_v2 = 0.73f;
    row.has_v1se = false;      // single-model fallback
    row.has_motion = false;    // window not filled yet
    row.has_ear = false;       // FaceMesh skipped this frame
    row.blink = false;
    row.face_px_size = 118.f;
    row.track_id = 7;
    row.ts = 1788680000.125;
    const std::string s = LivenessCaptureWriter::formatRow(row, "real");
    CHECK(s.find("\"P_tex_v1se\":null") != std::string::npos);
    CHECK(s.find("\"motion_residual\":null") != std::string::npos);
    CHECK(s.find("\"correlation\":null") != std::string::npos);
    CHECK(s.find("\"EAR\":null") != std::string::npos);
    CHECK(s.find("\"track_id\":7") != std::string::npos);
    CHECK(s.find("\"label\":\"real\"") != std::string::npos);
    CHECK(s.find("\"ts\":1788680000.125") != std::string::npos);
    CHECK(s.back() == '\n');
    // Never records pixels, embeddings or names.
    CHECK(s.find("name") == std::string::npos);

    const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                             "/liveness_capture_rows.jsonl";
    std::remove(path.c_str());
    LivenessCaptureWriter w;
    std::string err;
    CHECK(!w.append(row, &err));   // appending while stopped is an error
    CHECK(w.start(path, "real", 2, 60, 1000, &err));
    for (int i = 0; i < 3; ++i) CHECK(w.append(row, &err));
    CHECK(w.rows() == 3);
    CHECK(!w.expired(2000));
    CHECK(w.expired(3000));
    w.stop();

    std::ifstream in(path);
    int lines = 0;
    std::string line;
    while (std::getline(in, line)) ++lines;
    CHECK(lines == 3);

    // Append, never truncate: a second session adds to the same file.
    CHECK(w.start(path, "screen", 2, 60, 0, &err));
    CHECK(w.append(row, &err));
    w.stop();
    std::ifstream in2(path);
    lines = 0;
    std::string last;
    while (std::getline(in2, line)) {
        if (!line.empty()) last = line;
        ++lines;
    }
    CHECK(lines == 4);
    CHECK(last.find("\"label\":\"screen\"") != std::string::npos);
    std::remove(path.c_str());
}

int main() {
    run("motion_rigid_is_removed", test_motion_rigid_is_removed);
    run("motion_needs_window", test_motion_needs_window);
    run("motion_uncorrelated_jitter_scores_low", test_motion_uncorrelated_jitter_scores_low);
    run("motion_structured_deformation_scores_high",
        test_motion_structured_deformation_scores_high);
    run("motion_window_eviction_and_time_gap", test_motion_window_eviction_and_time_gap);
    run("blink_short_closures", test_blink_short_closures);
    run("blink_overlong_closure_is_not_a_blink", test_blink_overlong_closure_is_not_a_blink);
    run("blink_missing_samples_do_not_advance", test_blink_missing_samples_do_not_advance);
    run("fuse_pending_until_three_samples", test_fuse_pending_until_three_samples);
    run("fuse_ema", test_fuse_ema);
    run("fuse_blink_overrides_weak_texture", test_fuse_blink_overrides_weak_texture);
    run("fuse_no_blink_is_not_a_spoof", test_fuse_no_blink_is_not_a_spoof);
    run("fuse_texture_timeout_spoof", test_fuse_texture_timeout_spoof);
    run("fuse_hysteresis", test_fuse_hysteresis);
    run("fuse_motion_below_floor_is_ignored", test_fuse_motion_below_floor_is_ignored);
    run("fuse_motion_weighted", test_fuse_motion_weighted);
    run("track_retirement_is_a_fresh_state", test_track_retirement_is_a_fresh_state);
    run("capture_label_and_bounds", test_capture_label_and_bounds);
    run("capture_row_nulls_and_append", test_capture_row_nulls_and_append);

    std::cout << (g_failures ? "FAILED " : "PASSED ") << (g_checks - g_failures) << "/"
              << g_checks << " checks" << std::endl;
    return g_failures ? 1 : 0;
}
