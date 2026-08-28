#include "face_tracker.h"

#include <algorithm>
#include <numeric>

namespace face_analysis {

namespace {

float iouNorm(float ax, float ay, float aw, float ah,
              float bx, float by, float bw, float bh) {
    const float ix1 = std::max(ax, bx);
    const float iy1 = std::max(ay, by);
    const float ix2 = std::min(ax + aw, bx + bw);
    const float iy2 = std::min(ay + ah, by + bh);
    const float iw  = std::max(0.f, ix2 - ix1);
    const float ih  = std::max(0.f, iy2 - iy1);
    const float inter = iw * ih;
    const float uni   = aw * ah + bw * bh - inter;
    return uni > 1e-9f ? inter / uni : 0.f;
}

}  // namespace

FaceTracker::FaceTracker(const FaceTrackerConfig& cfg) : cfg_(cfg) {}

std::vector<int> FaceTracker::update(const std::vector<FaceInfo>& faces) {
    removed_ids_.clear();

    std::vector<int> assigned(faces.size(), -1);
    std::vector<char> det_taken(faces.size(), 0);

    // Older tracks get first pick. Two faces that overlap enough to be
    // ambiguous should stay with whoever has held the identity longest,
    // rather than with whichever track happens to sit earlier in the vector.
    std::vector<size_t> order(tracks_.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [this](size_t a, size_t b) {
        return tracks_[a].frames_tracked > tracks_[b].frames_tracked;
    });

    for (size_t oi = 0; oi < order.size(); ++oi) {
        Track& t = tracks_[order[oi]];
        int   best_det = -1;
        float best_iou = cfg_.iou_threshold;
        for (size_t d = 0; d < faces.size(); ++d) {
            if (det_taken[d]) continue;
            const float iou = iouNorm(t.x, t.y, t.w, t.h,
                                      faces[d].x, faces[d].y, faces[d].w, faces[d].h);
            if (iou > best_iou) {
                best_iou = iou;
                best_det = static_cast<int>(d);
            }
        }
        if (best_det >= 0) {
            det_taken[best_det] = 1;
            assigned[best_det]  = t.id;
            t.x = faces[best_det].x;
            t.y = faces[best_det].y;
            t.w = faces[best_det].w;
            t.h = faces[best_det].h;
            t.frames_tracked++;
            t.lost_frames = 0;
        } else {
            t.lost_frames++;
        }
    }

    // Retire the tracks that have been missing too long.
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (it->lost_frames > cfg_.max_lost_frames) {
            removed_ids_.push_back(it->id);
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }

    // Anything left unmatched is a new identity.
    for (size_t d = 0; d < faces.size(); ++d) {
        if (det_taken[d]) continue;
        Track t;
        t.id             = next_id_++;
        t.x              = faces[d].x;
        t.y              = faces[d].y;
        t.w              = faces[d].w;
        t.h              = faces[d].h;
        t.frames_tracked = 1;
        t.lost_frames    = 0;
        tracks_.push_back(t);
        assigned[d] = t.id;
    }

    return assigned;
}

int FaceTracker::trackFrames(int track_id) const {
    for (const auto& t : tracks_) {
        if (t.id == track_id) return t.frames_tracked;
    }
    return 0;
}

}  // namespace face_analysis
