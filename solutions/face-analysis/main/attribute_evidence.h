#ifndef _ATTRIBUTE_EVIDENCE_H_
#define _ATTRIBUTE_EVIDENCE_H_

#include <cstddef>
#include <vector>

#include "face_detector.h"

namespace face_analysis {

struct EvidenceConfig {
    float min_face_px      = 64.f;  // Quality gate: face box short side, in source-image pixels
    int   min_track_frames = 3;     // Gated-through frames needed before a verdict is "stable"
    float decay            = 1.0f;  // Per-frame decay on accumulated evidence; 1.0 = plain sum
};

// Quality gate. `face` is normalized; fw/fh are the source frame dimensions.
//
// The pre-existing `w < 0.01f` check in FaceDetector is a crop-safety guard
// (1% of frame), not an attribute-quality gate: a 30px face upsampled to the
// classifier input is mostly interpolation, and the heads still emit a
// confident-looking label for it.
bool passesGate(const FaceInfo& face, int fw, int fh, const EvidenceConfig& cfg);

struct Verdict {
    int   index      = -1;
    float confidence = 0.f;   // Vote share of the winner: its slice of the accumulated mass
    int   frames     = 0;
    bool  stable     = false;
};

// Per-track accumulation of per-head probability vectors.
//
// Semantics match the reCamera Pro implementation (kit/logic/attributes.py):
//   add()     -> sums = sums * decay + probs
//   verdict() -> argmax(sums), confidence = sums[argmax] / sum(sums)
//   stable    -> frames >= min_track_frames
class AttributeEvidence {
public:
    explicit AttributeEvidence(const EvidenceConfig& cfg = EvidenceConfig());

    // Fixed enum rather than string keys: this runs per face per frame on a
    // C906, and a map<string, vector<float>> would hash and allocate for it.
    enum Head { HEAD_RACE = 0, HEAD_GENDER, HEAD_AGE, HEAD_EMOTION, HEAD_COUNT };

    void    add(int track_id, Head head, const float* probs, int n);
    void    bumpFrame(int track_id);          // Once per track per frame, after all add() calls
    Verdict verdict(int track_id, Head head) const;
    void    sweep(const std::vector<int>& removed_ids);   // Reclaim retired tracks

    // Copies the accumulated sums, normalized to sum 1, into `out`.
    // Returns the number of values written (0 if the track/head has no evidence).
    int     shares(int track_id, Head head, float* out, int n) const;

    void    setConfig(const EvidenceConfig& cfg) { cfg_ = cfg; }
    const EvidenceConfig& config() const { return cfg_; }

    size_t  trackCount() const { return entries_.size(); }
    bool    hasTrack(int track_id) const { return find(track_id) != nullptr; }

private:
    // 7 + 2 + 9 + 8 = 26 floats per track, fixed length, no per-track allocation.
    struct Evidence {
        int   track_id = -1;
        int   frames   = 0;
        float race[7]    = {};
        float gender[2]  = {};
        float age[9]     = {};
        float emotion[8] = {};
    };

    static float*       headPtr(Evidence& e, Head head, int& n);
    static const float* headPtr(const Evidence& e, Head head, int& n);

    Evidence*       find(int track_id);
    const Evidence* find(int track_id) const;
    Evidence&       findOrCreate(int track_id);

    EvidenceConfig        cfg_;
    std::vector<Evidence> entries_;
};

}  // namespace face_analysis

#endif  // _ATTRIBUTE_EVIDENCE_H_
