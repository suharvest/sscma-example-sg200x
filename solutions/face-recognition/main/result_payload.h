#ifndef _FACE_RECOGNITION_RESULT_PAYLOAD_H_
#define _FACE_RECOGNITION_RESULT_PAYLOAD_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "face_types.h"

// Deliberately no nlohmann type in this header. main.cpp pulls in libhv through
// <sscma.h>, and the SDK ships its own older copy of nlohmann/json under
// hv/json.hpp -- so a `nlohmann::json` in a signature mangles differently in
// main.cpp than in this translation unit and the link fails on a symbol that
// looks identical in the error message. Everything crossing this boundary is a
// serialized string.

namespace face_recognition {

// Small builder for the free-form `data` half of an acknowledgement, so callers
// never need a JSON type of their own.
class AckData {
public:
    void addString(const std::string& key, const std::string& value);
    void addUInt(const std::string& key, uint64_t value);
    // `raw_json` is spliced in verbatim; use for arrays and nested objects.
    void addRaw(const std::string& key, const std::string& raw_json);

    bool empty() const { return fields_.empty(); }
    // "{...}" of everything added so far, or "{}".
    std::string dump() const;

private:
    std::vector<std::pair<std::string, std::string>> fields_;  // key -> serialized value
};

// Results envelope published to recamera/face-recognition/results.
//
// Field names match the reCamera Pro face-recognition app exactly, so one
// consumer reads either device: faces[] carries
// {track_id, bbox (normalized xyxy), det_score, name, score, live,
//  liveness_score, stable, gated, reason}.
std::string buildResultsJson(uint64_t timestamp_ms, uint32_t frame_id,
                             float inference_time_ms,
                             int source_width, int source_height,
                             const std::string& model_tag,
                             const std::vector<RecognitionResult>& faces);

// Acknowledgement published to recamera/face-recognition/cmd/ack.
std::string buildAckJson(const std::string& request_id, const std::string& op,
                         bool ok, const std::string& err,
                         const std::string& model_tag, size_t users,
                         const AckData& data = AckData());

// `[{"name":...,"n":...}, ...]` for the `list` acknowledgement.
std::string buildNamesJson(const std::vector<std::pair<std::string, uint32_t>>& users);

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_RESULT_PAYLOAD_H_
