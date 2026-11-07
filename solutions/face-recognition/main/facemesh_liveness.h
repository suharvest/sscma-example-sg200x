#ifndef _FACE_RECOGNITION_FACEMESH_LIVENESS_H_
#define _FACE_RECOGNITION_FACEMESH_LIVENESS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "face_types.h"
#include "facemesh_landmarker.h"

namespace face_recognition {

// Blink evidence for liveness: one MediaPipe FaceMesh inference per sampled
// frame, reduced to a single average EAR.
//
// What is reused, and what is not
// -------------------------------
// The landmark model runner (`facemesh_reader::FacemeshLandmarker`) and the EAR
// maths (`facemesh_reader::computeMetrics`) are compiled in from
// solutions/facemesh-reader/main/ verbatim. The crop/mapping is reimplemented
// here rather than reusing `FacemeshPipeline` because that class also owns a
// YOLO-Face detector, a drowsiness state machine and a yawn detector; pulling it
// in would drag a second face detector into a process whose binding constraint
// is the model ION pool. The geometry below is the same as
// facemesh_pipeline.cpp:20-97, with one difference stated explicitly:
//
//   BBOX CONVENTION. FacemeshPipeline takes a `FaceInfo` whose (x, y) the
//   surrounding docs describe inconsistently as centre and as top-left. The CODE
//   is unambiguous -- facemesh_pipeline.cpp computes
//       x1 = face.x * fw ; x2 = (face.x + face.w) * fw
//   so (x, y) is the TOP-LEFT corner, normalized, and (w, h) is the size. This
//   class sidesteps the ambiguity entirely by taking a face_recognition::BBox,
//   which is documented as source-pixel corners and has no second reading.
//
// Only the average EAR is returned. Blink STATE (consecutive closures, the
// latch) belongs to the recognition track and lives in LivenessState, because
// one FacemeshLiveness instance serves every track in the frame.
class FacemeshLiveness {
public:
    // Optional in the same sense as the texture stage: a failure disables blink
    // evidence and leaves texture and motion running.
    bool initOptional(const std::string& model_path, std::string* error = nullptr);
    bool available() const { return ready_; }
    std::string inputDescription() const;

    // `rgb` is packed RGB888 of the inference frame. Returns the average of the
    // left and right six-point EARs.
    bool infer(const uint8_t* rgb, int width, int height, const BBox& bbox, float& ear,
               std::string* error = nullptr);

    void setBboxPadding(float pad) { bbox_padding_ = pad; }

private:
    facemesh_reader::FacemeshLandmarker landmarker_;
    bool ready_ = false;
    int size_ = 192;
    float bbox_padding_ = 0.10f;
    std::vector<uint8_t> roi_;
    std::vector<facemesh_reader::Point2D> mapped_;
};

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_FACEMESH_LIVENESS_H_
