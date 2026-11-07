#ifndef _FACE_RECOGNITION_LIVENESS_CAPTURE_WRITER_H_
#define _FACE_RECOGNITION_LIVENESS_CAPTURE_WRITER_H_

#include <cstdint>
#include <cstdio>
#include <string>

// Append-only JSONL sink for liveness calibration capture.
//
// Records FEATURES, never pixels: no image, no embedding, no gallery name ever
// reaches this file. That is what makes it safe to leave on a device and pull
// off for offline threshold fitting.
//
// Device dependency free (stdio only) so the host test target can exercise it.

namespace face_recognition {

struct CaptureRow {
    bool has_v2 = false;
    float p_tex_v2 = 0.f;
    bool has_v1se = false;
    float p_tex_v1se = 0.f;
    bool has_motion = false;
    float motion_residual = 0.f;
    float correlation = 0.f;
    bool has_ear = false;
    float ear = 0.f;
    bool blink = false;
    float face_px_size = 0.f;
    int track_id = -1;
    double ts = 0.0;  // wall-clock seconds
};

class LivenessCaptureWriter {
public:
    ~LivenessCaptureWriter() { stop(); }

    // `seconds` is clamped to [1, max_seconds]. `label` must be one of
    // real|print|screen; anything else is rejected here rather than written into
    // a file that a fitter would later trust.
    bool start(const std::string& path, const std::string& label, int seconds,
               int max_seconds, uint64_t now_ms, std::string* error);

    bool active() const { return file_ != nullptr; }
    bool expired(uint64_t now_ms) const { return active() && now_ms >= deadline_ms_; }

    bool append(const CaptureRow& row, std::string* error);

    // Flushes at most once per second; call every frame.
    void tick(uint64_t now_ms);

    // Closes the file (flushing first). Safe to call when inactive.
    void stop();

    static bool validLabel(const std::string& label);

    const std::string& path() const { return path_; }
    const std::string& label() const { return label_; }
    uint64_t rows() const { return rows_; }
    int seconds() const { return seconds_; }

    // Serializes one row; exposed so the tests can assert on the exact JSON
    // without touching the filesystem.
    static std::string formatRow(const CaptureRow& row, const std::string& label);

private:
    std::FILE* file_ = nullptr;
    std::string path_;
    std::string label_;
    uint64_t deadline_ms_ = 0;
    uint64_t last_flush_ms_ = 0;
    uint64_t rows_ = 0;
    int seconds_ = 0;
};

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_LIVENESS_CAPTURE_WRITER_H_
