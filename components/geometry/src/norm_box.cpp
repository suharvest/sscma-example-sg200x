#include "norm_box.h"

#include <type_traits>

namespace geometry {

/*
 * The properties the design rests on, checked on every build rather than
 * asserted in a comment. If someone "simplifies" InferBox and StreamBox into
 * one type, or gives NormBox a converting constructor, this is what says so --
 * at which point handing an inference-normalised box to the mask compiles again
 * and the masks quietly go back to covering three quarters of a face.
 *
 * The negative case cannot be written here (code that must NOT compile), so for
 * the record: passing std::vector<InferBox> to a std::vector<StreamBox>
 * parameter fails with "no known conversion", which is the whole point.
 */
static_assert(!std::is_convertible<InferBox, StreamBox>::value,
              "the two frames must not silently convert");
static_assert(!std::is_convertible<StreamBox, InferBox>::value,
              "the two frames must not silently convert");
static_assert(!std::is_same<InferBox, StreamBox>::value,
              "the two frames must stay distinct types");
static_assert(sizeof(InferBox) == 5 * sizeof(float),
              "still four floats and a score -- the safety is compile-time only");

std::vector<StreamBox> toStream(const std::vector<InferBox>& boxes,
                                int inference_w, int inference_h,
                                int stream_w, int stream_h) {
    std::vector<StreamBox> out;
    out.reserve(boxes.size());

    if (inference_w <= 0 || inference_h <= 0 || stream_w <= 0 || stream_h <= 0) {
        /* Nothing sensible to convert against. Returning the boxes unconverted
         * would put a mask somewhere arbitrary; returning none leaves the frame
         * unmasked, which is visible. Prefer the visible failure. */
        return out;
    }

    /* Where the scene sits inside the inference frame, as a fraction of that
     * frame. Cross-multiply rather than divide so the aspect test is exact. */
    float content_w = 1.0f, content_h = 1.0f;
    float x_off = 0.0f, y_off = 0.0f;
    const long long stream_vs_inf =
        (long long)stream_w * inference_h - (long long)stream_h * inference_w;
    if (stream_vs_inf > 0) {
        /* Stream is the wider shape -> bars top and bottom. */
        content_h = ((float)inference_w * stream_h / stream_w) / (float)inference_h;
        y_off     = (1.0f - content_h) * 0.5f;
    } else if (stream_vs_inf < 0) {
        /* Stream is the taller shape -> bars left and right. */
        content_w = ((float)inference_h * stream_w / stream_h) / (float)inference_w;
        x_off     = (1.0f - content_w) * 0.5f;
    }
    /* else: same shape, the two normalised frames coincide -- the loop below
     * copies through, which is what changes the type. */

    for (const auto& b : boxes) {
        out.push_back(StreamBox::fromCenter((b.cx - x_off) / content_w,
                                            (b.cy - y_off) / content_h,
                                            b.w / content_w,
                                            b.h / content_h,
                                            b.score));
    }
    return out;
}

}  // namespace geometry
