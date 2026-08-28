#include "attribute_evidence.h"

#include <algorithm>
#include <cmath>

namespace face_analysis {

bool passesGate(const FaceInfo& face, int fw, int fh, const EvidenceConfig& cfg) {
    if (fw <= 0 || fh <= 0) return false;
    const float px_w = face.w * static_cast<float>(fw);
    const float px_h = face.h * static_cast<float>(fh);
    return std::min(px_w, px_h) >= cfg.min_face_px;
}

AttributeEvidence::AttributeEvidence(const EvidenceConfig& cfg) : cfg_(cfg) {}

float* AttributeEvidence::headPtr(Evidence& e, Head head, int& n) {
    switch (head) {
        case HEAD_RACE:    n = 7; return e.race;
        case HEAD_GENDER:  n = 2; return e.gender;
        case HEAD_AGE:     n = 9; return e.age;
        case HEAD_EMOTION: n = 8; return e.emotion;
        default:           n = 0; return nullptr;
    }
}

const float* AttributeEvidence::headPtr(const Evidence& e, Head head, int& n) {
    return headPtr(const_cast<Evidence&>(e), head, n);
}

AttributeEvidence::Evidence* AttributeEvidence::find(int track_id) {
    for (auto& e : entries_) {
        if (e.track_id == track_id) return &e;
    }
    return nullptr;
}

const AttributeEvidence::Evidence* AttributeEvidence::find(int track_id) const {
    for (const auto& e : entries_) {
        if (e.track_id == track_id) return &e;
    }
    return nullptr;
}

AttributeEvidence::Evidence& AttributeEvidence::findOrCreate(int track_id) {
    Evidence* e = find(track_id);
    if (e) return *e;
    Evidence fresh;
    fresh.track_id = track_id;
    entries_.push_back(fresh);
    return entries_.back();
}

void AttributeEvidence::add(int track_id, Head head, const float* probs, int n) {
    if (!probs || n <= 0) return;
    Evidence& e = findOrCreate(track_id);
    int cap = 0;
    float* dst = headPtr(e, head, cap);
    if (!dst || cap <= 0) return;
    const int m = std::min(n, cap);
    for (int i = 0; i < cap; ++i) {
        dst[i] = dst[i] * cfg_.decay + (i < m ? probs[i] : 0.f);
    }
}

void AttributeEvidence::bumpFrame(int track_id) {
    findOrCreate(track_id).frames++;
}

Verdict AttributeEvidence::verdict(int track_id, Head head) const {
    Verdict v;
    const Evidence* e = find(track_id);
    if (!e) return v;

    v.frames = e->frames;
    v.stable = (e->frames >= cfg_.min_track_frames);

    int cap = 0;
    const float* src = headPtr(*e, head, cap);
    if (!src || cap <= 0) return v;

    float total = 0.f;
    int   best  = -1;
    float bestv = 0.f;
    for (int i = 0; i < cap; ++i) {
        total += src[i];
        if (src[i] > bestv) {
            bestv = src[i];
            best  = i;
        }
    }
    if (best < 0 || total <= 0.f) return v;

    v.index      = best;
    v.confidence = bestv / total;
    return v;
}

int AttributeEvidence::shares(int track_id, Head head, float* out, int n) const {
    if (!out || n <= 0) return 0;
    const Evidence* e = find(track_id);
    if (!e) return 0;
    int cap = 0;
    const float* src = headPtr(*e, head, cap);
    if (!src || cap <= 0) return 0;
    float total = 0.f;
    for (int i = 0; i < cap; ++i) total += src[i];
    if (total <= 0.f) return 0;
    const int m = std::min(n, cap);
    for (int i = 0; i < m; ++i) out[i] = src[i] / total;
    return m;
}

void AttributeEvidence::sweep(const std::vector<int>& removed_ids) {
    if (removed_ids.empty()) return;
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [&removed_ids](const Evidence& e) {
                                      return std::find(removed_ids.begin(), removed_ids.end(),
                                                       e.track_id) != removed_ids.end();
                                  }),
                   entries_.end());
    // Actually give the memory back. Without this the vector keeps the
    // high-water capacity of every face that ever walked past the camera.
    if (entries_.capacity() > entries_.size() * 4 + 8) {
        std::vector<Evidence>(entries_).swap(entries_);
    }
}

}  // namespace face_analysis
