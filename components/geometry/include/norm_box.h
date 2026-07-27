#ifndef GEOMETRY_NORM_BOX_H
#define GEOMETRY_NORM_BOX_H

#include <vector>

/*
 * Normalised boxes that say what they are.
 *
 * Every detector in this tree emits four floats in [0,1]. Four floats cannot
 * answer either of the two questions a consumer has to get right:
 *
 *   1. Is x/y the CENTRE of the box, or its top-left corner?
 *   2. Normalised against WHICH frame -- the inference channel, or the video
 *      stream?
 *
 * Neither answer is uniform here. The face detector behind facemesh-reader and
 * yolo-detector passes the model's centre xy straight through; face-analysis's
 * does not. VPSS fits the scene into each channel preserving aspect, so a 4:3
 * inference channel and a 16:9 stream disagree about what y = 0.5 means.
 *
 * Both questions have been answered wrongly in this tree, and neither mistake
 * was loud: a mask half a box down-and-right of the face still looks like a
 * working privacy feature in a screenshot, and a mask 3/4 of the height it
 * should be only fails for people near the top and bottom of frame. Comments
 * warning about both existed. Comments do not participate in compilation.
 *
 * So the frame is a type parameter and the origin convention is a named
 * constructor. `PrivacyBlur::onDetection` takes a StreamBox and nothing else;
 * handing it an InferBox is a compile error rather than a subtly misplaced
 * mask. There is no aggregate initialisation, so writing a box down forces
 * whoever holds the detector output -- the only person who knows -- to say
 * which convention it uses.
 *
 * Zero runtime cost: same four floats, same layout, no virtuals.
 */
namespace geometry {

enum class Frame {
    Inference,  /* normalised against the inference channel (model input) */
    Stream,     /* normalised against the encoded video stream */
};

template <Frame F>
struct NormBox {
    /* Centre form throughout. The edges are computed, never stored, so there
     * is exactly one representation and no pair of fields to disagree. */
    float cx = 0.0f;
    float cy = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float score = 0.0f;

    static NormBox fromCenter(float cx, float cy, float w, float h, float score = 0.0f) {
        NormBox b;
        b.cx = cx;
        b.cy = cy;
        b.w = w;
        b.h = h;
        b.score = score;
        return b;
    }

    static NormBox fromCorner(float left, float top, float w, float h, float score = 0.0f) {
        NormBox b;
        b.cx = left + w * 0.5f;
        b.cy = top + h * 0.5f;
        b.w = w;
        b.h = h;
        b.score = score;
        return b;
    }

    float left() const { return cx - w * 0.5f; }
    float top() const { return cy - h * 0.5f; }
    float right() const { return cx + w * 0.5f; }
    float bottom() const { return cy + h * 0.5f; }
};

using InferBox  = NormBox<Frame::Inference>;
using StreamBox = NormBox<Frame::Stream>;

/*
 * The only way across the frame boundary.
 *
 * VPSS fits the sensor content into each channel preserving aspect, so the
 * scene sits inside the inference frame as a band with bars on two sides. This
 * maps a box out of that band and onto the stream. A no-op in value (but not in
 * type) when the two aspects match.
 *
 * Equivalent to debug_stream_letterbox_to_display(), which does the same
 * geometry in pixels for the console overlay; this one exists so an application
 * can convert without depending on the debug HTTP server.
 */
std::vector<StreamBox> toStream(const std::vector<InferBox>& boxes,
                                int inference_w, int inference_h,
                                int stream_w, int stream_h);

}  // namespace geometry

#endif  // GEOMETRY_NORM_BOX_H
