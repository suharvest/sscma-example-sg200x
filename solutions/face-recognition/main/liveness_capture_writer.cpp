#include "liveness_capture_writer.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>

namespace face_recognition {

namespace {

void setError(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}

// Numbers only; the label is the sole string in a row and is validated against a
// closed set, so no general JSON escaping is needed here.
std::string num(double v, int decimals) {
    if (!std::isfinite(v)) return "null";
    char buf[48];
    snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

}  // namespace

bool LivenessCaptureWriter::validLabel(const std::string& label) {
    return label == "real" || label == "print" || label == "screen";
}

bool LivenessCaptureWriter::start(const std::string& path, const std::string& label,
                                  int seconds, int max_seconds, uint64_t now_ms,
                                  std::string* error) {
    if (active()) {
        setError(error, "a liveness capture is already running");
        return false;
    }
    if (!validLabel(label)) {
        setError(error, "label must be one of real, print, screen");
        return false;
    }
    if (max_seconds < 1) max_seconds = 1;
    if (seconds < 1) seconds = 1;
    if (seconds > max_seconds) seconds = max_seconds;

    // Append, never truncate: several sessions of the same label are the point.
    file_ = std::fopen(path.c_str(), "ae");
    if (!file_) {
        setError(error, std::string("cannot open ") + path + ": " + std::strerror(errno));
        return false;
    }
    path_ = path;
    label_ = label;
    seconds_ = seconds;
    deadline_ms_ = now_ms + static_cast<uint64_t>(seconds) * 1000ull;
    last_flush_ms_ = now_ms;
    rows_ = 0;
    return true;
}

std::string LivenessCaptureWriter::formatRow(const CaptureRow& row, const std::string& label) {
    std::string s = "{";
    s += "\"P_tex_v2\":" + (row.has_v2 ? num(row.p_tex_v2, 6) : std::string("null"));
    s += ",\"P_tex_v1se\":" + (row.has_v1se ? num(row.p_tex_v1se, 6) : std::string("null"));
    s += ",\"motion_residual\":" +
         (row.has_motion ? num(row.motion_residual, 6) : std::string("null"));
    s += ",\"correlation\":" + (row.has_motion ? num(row.correlation, 6) : std::string("null"));
    s += ",\"EAR\":" + (row.has_ear ? num(row.ear, 6) : std::string("null"));
    s += ",\"blink\":";
    s += row.blink ? "true" : "false";
    s += ",\"face_px_size\":" + num(row.face_px_size, 2);
    s += ",\"track_id\":" + std::to_string(row.track_id);
    s += ",\"label\":\"" + label + "\"";
    s += ",\"ts\":" + num(row.ts, 3);
    s += "}\n";
    return s;
}

bool LivenessCaptureWriter::append(const CaptureRow& row, std::string* error) {
    if (!active()) {
        setError(error, "no liveness capture is running");
        return false;
    }
    const std::string line = formatRow(row, label_);
    if (std::fwrite(line.data(), 1, line.size(), file_) != line.size()) {
        setError(error, "short write to capture file");
        return false;
    }
    rows_++;
    return true;
}

void LivenessCaptureWriter::tick(uint64_t now_ms) {
    if (!active()) return;
    if (now_ms - last_flush_ms_ >= 1000ull) {
        std::fflush(file_);
        last_flush_ms_ = now_ms;
    }
}

void LivenessCaptureWriter::stop() {
    if (!file_) return;
    std::fflush(file_);
    std::fclose(file_);
    file_ = nullptr;
}

}  // namespace face_recognition
