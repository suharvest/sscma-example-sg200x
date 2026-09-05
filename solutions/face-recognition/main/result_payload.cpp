#include "result_payload.h"

#include <algorithm>
#include <cmath>

#include "json.hpp"

namespace face_recognition {

namespace {

double round3(float v) { return std::round(static_cast<double>(v) * 1000.0) / 1000.0; }

double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

}  // namespace

std::string buildResultsJson(uint64_t timestamp_ms, uint32_t frame_id,
                             float inference_time_ms,
                             int source_width, int source_height,
                             const std::string& model_tag,
                             const std::vector<RecognitionResult>& faces) {
    const double w = source_width > 0 ? source_width : 1;
    const double h = source_height > 0 ? source_height : 1;

    nlohmann::json rows = nlohmann::json::array();
    for (const auto& f : faces) {
        nlohmann::json row;
        row["track_id"] = f.track_id;
        // Normalized xyxy against the inference frame, same as the Pro app.
        row["bbox"] = nlohmann::json::array({
            std::round(clamp01(f.bbox.x1 / w) * 1e4) / 1e4,
            std::round(clamp01(f.bbox.y1 / h) * 1e4) / 1e4,
            std::round(clamp01(f.bbox.x2 / w) * 1e4) / 1e4,
            std::round(clamp01(f.bbox.y2 / h) * 1e4) / 1e4,
        });
        row["det_score"] = round3(f.det_score);
        if (f.name.empty()) {
            row["name"] = nullptr;
        } else {
            row["name"] = f.name;
        }
        row["score"] = round3(f.score);
        // live/liveness_score stay null when the stage did not run, so a
        // consumer can tell "not checked" from "checked and failed".
        if (f.has_liveness) {
            row["live"] = f.live;
            row["liveness_score"] = round3(f.liveness_score);
        } else {
            row["live"] = nullptr;
            row["liveness_score"] = nullptr;
        }
        row["stable"] = f.stable;
        row["gated"] = f.gated;
        if (f.reason.empty()) {
            row["reason"] = nullptr;
        } else {
            row["reason"] = f.reason;
        }
        rows.push_back(std::move(row));
    }

    nlohmann::json out;
    out["timestamp"] = timestamp_ms;
    out["frame_id"] = frame_id;
    out["inference_time_ms"] = round3(inference_time_ms);
    out["model_tag"] = model_tag;
    out["face_count"] = faces.size();
    out["faces"] = std::move(rows);
    return out.dump();
}

void AckData::addString(const std::string& key, const std::string& value) {
    fields_.emplace_back(key, nlohmann::json(value).dump());
}

void AckData::addUInt(const std::string& key, uint64_t value) {
    fields_.emplace_back(key, std::to_string(value));
}

void AckData::addRaw(const std::string& key, const std::string& raw_json) {
    fields_.emplace_back(key, raw_json);
}

std::string AckData::dump() const {
    std::string out = "{";
    for (size_t i = 0; i < fields_.size(); ++i) {
        if (i) out += ",";
        out += nlohmann::json(fields_[i].first).dump();
        out += ":";
        out += fields_[i].second;
    }
    out += "}";
    return out;
}

std::string buildAckJson(const std::string& request_id, const std::string& op,
                         bool ok, const std::string& err,
                         const std::string& model_tag, size_t users,
                         const AckData& data) {
    nlohmann::json out;
    if (request_id.empty()) {
        out["request_id"] = nullptr;
    } else {
        out["request_id"] = request_id;
    }
    out["op"] = op;
    out["ok"] = ok;
    out["model_tag"] = model_tag;
    out["users"] = users;
    if (!ok) out["err"] = err;
    if (!data.empty()) {
        const nlohmann::json extra = nlohmann::json::parse(data.dump(), nullptr, false);
        if (extra.is_object()) {
            for (auto it = extra.begin(); it != extra.end(); ++it) out[it.key()] = it.value();
        }
    }
    return out.dump();
}

std::string buildNamesJson(const std::vector<std::pair<std::string, uint32_t>>& users) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& kv : users) {
        arr.push_back({{"name", kv.first}, {"n", kv.second}});
    }
    return arr.dump();
}

}  // namespace face_recognition
