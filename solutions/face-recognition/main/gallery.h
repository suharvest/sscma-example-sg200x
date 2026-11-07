#ifndef _FACE_RECOGNITION_GALLERY_H_
#define _FACE_RECOGNITION_GALLERY_H_

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "face_types.h"

namespace face_recognition {

struct GalleryRecord {
    Embedding512 emb{};
    uint32_t n = 0;
    std::string ts;
};

struct GalleryMatch {
    std::string name;
    float score = 0.f;   // best cosine observed, whether or not it matched
    bool matched = false;
};

// One averaged 512-D template per person, persisted as JSON v2:
//
//   {"_meta": {"model_tag","dim","threshold","updated","format":2},
//    "users": {"<name>": {"emb":[512 floats], "n":int, "ts":iso8601}}}
//
// The model tag is a hard gate. ArcFace weights differ per backend, so an
// embedding enrolled against one model and compared against another scores
// cosine ~0 -- which does not fail, it just recognises nobody. Loading a
// gallery whose `_meta.model_tag` disagrees is therefore an error, not a
// warning, and the file is left untouched.
class Gallery {
public:
    bool load(const std::string& path, const std::string& expected_model_tag,
              float threshold, std::string* error = nullptr);
    bool reload(std::string* error = nullptr);

    GalleryMatch match(const Embedding512& query) const;

    // Averages `embeddings`, re-normalizes and stores under `name`, replacing
    // any prior record, then rewrites the file atomically.
    bool enroll(const std::string& name, const std::vector<Embedding512>& embeddings,
                std::string* error = nullptr);
    bool remove(const std::string& name, std::string* error = nullptr);

    std::vector<std::pair<std::string, uint32_t>> list() const;
    size_t size() const { return users_.size(); }
    const std::string& path() const { return path_; }
    const std::string& modelTag() const { return model_tag_; }
    float threshold() const { return threshold_; }
    void setThreshold(float v) { threshold_ = v; }

    static size_t maxUsers() { return 1000; }

private:
    bool save(std::string* error);

    std::string path_;
    std::string model_tag_;
    float threshold_ = 0.40f;
    std::map<std::string, GalleryRecord> users_;
};

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_GALLERY_H_
