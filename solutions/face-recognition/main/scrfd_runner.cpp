#include "scrfd_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define TAG "face-recognition/scrfd"

namespace face_recognition {

namespace {

constexpr int kStrides[3] = {8, 16, 32};
constexpr int kNumAnchors = 2;
constexpr uint8_t kPadValue = 128;  // matches the fp32 reference letterbox

void setError(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}

int kindForCols(int cols) {
    switch (cols) {
        case 1: return 0;   // scores
        case 4: return 1;   // bbox deltas
        case 10: return 2;  // 5-point keypoint deltas
        default: return -1;
    }
}

const char* kindName(int k) {
    static const char* n[3] = {"scores", "bboxes", "kps"};
    return (k >= 0 && k < 3) ? n[k] : "?";
}

}  // namespace

bool ScrfdRunner::init(const std::string& model_path, std::string* error) {
    ready_ = false;
    // The int8 converter baked mean 127.5 / scale 1/128; on an S8 input tensor
    // CviRunner turns that into the `pixel - 128` passthrough the runtime wants.
    // The mean/scale below are only consulted if the model is ever re-converted
    // to a float input.
    const float mean[3] = {127.5f, 127.5f, 127.5f};
    const float scale[3] = {1.f / 128.f, 1.f / 128.f, 1.f / 128.f};
    if (!runner_.init(model_path, mean, scale, /*bgr=*/false, error)) return false;

    if (runner_.inputWidth() != runner_.inputHeight()) {
        setError(error, "SCRFD input is not square");
        return false;
    }
    input_size_ = runner_.inputWidth();
    if (input_size_ != 640) {
        setError(error, "SCRFD input is " + std::to_string(input_size_) +
                            ", this decoder is built for 640");
        return false;
    }
    if (runner_.inputChannels() != 3) {
        setError(error, "SCRFD input is not 3-channel");
        return false;
    }
    if (!mapOutputs(error)) return false;

    letterbox_.assign(static_cast<size_t>(input_size_) * input_size_ * 3u, kPadValue);
    ready_ = true;
    return true;
}

bool ScrfdRunner::mapOutputs(std::string* error) {
    const int n = runner_.outputCount();
    if (n != 9) {
        setError(error, "SCRFD expects 9 outputs, model has " + std::to_string(n));
        return false;
    }

    std::ostringstream desc;
    std::vector<HeadRef> refs;
    refs.reserve(9);

    for (int i = 0; i < n; ++i) {
        const ma_tensor_t t = runner_.output(i);
        const ma_shape_t& s = t.shape;
        if (i) desc << " ";
        desc << "#" << i << CviRunner::shapeToString(s);

        // Squeeze the unit dimensions the CVI runtime pads every tensor out to.
        std::vector<int> v;
        for (int d = 0; d < s.size; ++d) {
            if (s.dims[d] > 1) v.push_back(s.dims[d]);
        }

        HeadRef h;
        h.tensor_index = i;
        if (v.size() == 1) {
            h.rows = v[0];
            h.cols = 1;
        } else if (v.size() == 2) {
            if (kindForCols(v[1]) >= 0) {
                h.rows = v[0];
                h.cols = v[1];
            } else if (kindForCols(v[0]) >= 0) {
                h.rows = v[1];
                h.cols = v[0];
            } else {
                setError(error, "SCRFD output #" + std::to_string(i) +
                                    " has no recognizable column count " +
                                    CviRunner::shapeToString(s));
                return false;
            }
        } else if (v.size() == 3) {
            // (C,H,W) with C = kNumAnchors * cols.
            const int C = v[0];
            if (C % kNumAnchors != 0 || kindForCols(C / kNumAnchors) < 0) {
                setError(error, "SCRFD output #" + std::to_string(i) +
                                    " channel count " + std::to_string(C) +
                                    " is not 2/8/20");
                return false;
            }
            h.nchw = true;
            h.H = v[1];
            h.W = v[2];
            h.cols = C / kNumAnchors;
            h.rows = h.H * h.W * kNumAnchors;
        } else {
            setError(error, "SCRFD output #" + std::to_string(i) +
                                " has unexpected rank " +
                                CviRunner::shapeToString(s));
            return false;
        }
        refs.push_back(h);
    }
    output_desc_ = desc.str();

    for (int si = 0; si < 3; ++si) {
        for (int k = 0; k < 3; ++k) heads_[si][k] = HeadRef{};
    }

    for (const auto& h : refs) {
        int si = -1;
        for (int i = 0; i < 3; ++i) {
            const int side = input_size_ / kStrides[i];
            if (h.rows == side * side * kNumAnchors) {
                si = i;
                break;
            }
        }
        if (si < 0) {
            setError(error, "SCRFD output #" + std::to_string(h.tensor_index) +
                                " row count " + std::to_string(h.rows) +
                                " matches no stride");
            return false;
        }
        const int k = kindForCols(h.cols);
        if (heads_[si][k].tensor_index >= 0) {
            setError(error, std::string("SCRFD has two '") + kindName(k) +
                                "' heads for stride " + std::to_string(kStrides[si]));
            return false;
        }
        heads_[si][k] = h;
    }

    for (int si = 0; si < 3; ++si) {
        for (int k = 0; k < 3; ++k) {
            if (heads_[si][k].tensor_index < 0) {
                setError(error, std::string("SCRFD stride ") +
                                    std::to_string(kStrides[si]) + " is missing its " +
                                    kindName(k) + " head");
                return false;
            }
        }
    }
    return true;
}

float ScrfdRunner::headValue(const ma_tensor_t& t, const HeadRef& h, int row,
                             int col) const {
    size_t idx;
    if (h.nchw) {
        // row = (y * W + x) * anchors + anchor ; channel = anchor * cols + col
        const int anchor = row % kNumAnchors;
        const int cell = row / kNumAnchors;
        const int c = anchor * h.cols + col;
        idx = static_cast<size_t>(c) * h.H * h.W + static_cast<size_t>(cell);
    } else {
        idx = static_cast<size_t>(row) * h.cols + static_cast<size_t>(col);
    }
    return CviRunner::readValue(t, idx);
}

bool ScrfdRunner::infer(const uint8_t* rgb, int width, int height,
                        std::vector<FaceDetection>& out, std::string* error) {
    out.clear();
    if (!ready_ || !rgb || width <= 0 || height <= 0) {
        setError(error, "SCRFD runner not ready");
        return false;
    }

    // Letterbox: aspect preserved, centred, grey padding.
    const float scale = std::min(static_cast<float>(input_size_) / width,
                                 static_cast<float>(input_size_) / height);
    const int nw = std::max(1, static_cast<int>(std::lround(width * scale)));
    const int nh = std::max(1, static_cast<int>(std::lround(height * scale)));
    const int pad_w = (input_size_ - nw) / 2;
    const int pad_h = (input_size_ - nh) / 2;

    const auto t_lb0 = std::chrono::steady_clock::now();
    std::fill(letterbox_.begin(), letterbox_.end(), kPadValue);
    {
        const cv::Mat src(height, width, CV_8UC3, const_cast<uint8_t*>(rgb));
        cv::Mat canvas(input_size_, input_size_, CV_8UC3, letterbox_.data());
        cv::Mat roi = canvas(cv::Rect(pad_w, pad_h, nw, nh));
        cv::resize(src, roi, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    }
    const auto t_lb1 = std::chrono::steady_clock::now();

    if (!runner_.run(letterbox_.data(), error)) return false;
    const auto t_run1 = std::chrono::steady_clock::now();

    struct Prop {
        float x1, y1, x2, y2, score;
        float kps[10];
    };
    std::vector<Prop> props;

    for (int si = 0; si < 3; ++si) {
        const int stride = kStrides[si];
        const int side = input_size_ / stride;
        const ma_tensor_t ts = runner_.output(heads_[si][0].tensor_index);
        const ma_tensor_t tb = runner_.output(heads_[si][1].tensor_index);
        const ma_tensor_t tk = runner_.output(heads_[si][2].tensor_index);
        const HeadRef& hs = heads_[si][0];
        const HeadRef& hb = heads_[si][1];
        const HeadRef& hk = heads_[si][2];

        for (int row = 0; row < hs.rows; ++row) {
            const float score = headValue(ts, hs, row, 0);
            if (score < score_threshold_) continue;

            // InsightFace anchor convention: centre = grid index * stride, with
            // NO half-cell offset, and both anchors of a cell share the centre.
            const int cell = row / kNumAnchors;
            const int gx = cell % side;
            const int gy = cell / side;
            const float cx = static_cast<float>(gx) * stride;
            const float cy = static_cast<float>(gy) * stride;

            Prop p;
            p.score = std::min(1.0f, std::max(0.0f, score));
            p.x1 = cx - headValue(tb, hb, row, 0) * stride;
            p.y1 = cy - headValue(tb, hb, row, 1) * stride;
            p.x2 = cx + headValue(tb, hb, row, 2) * stride;
            p.y2 = cy + headValue(tb, hb, row, 3) * stride;
            for (int i = 0; i < 5; ++i) {
                p.kps[i * 2] = cx + headValue(tk, hk, row, i * 2) * stride;
                p.kps[i * 2 + 1] = cy + headValue(tk, hk, row, i * 2 + 1) * stride;
            }
            props.push_back(p);
        }
    }

    if (!props.empty()) {
    // One global score-ordered IoU NMS across all three strides.
    std::vector<int> order(props.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return props[a].score > props[b].score;
    });

    std::vector<char> suppressed(props.size(), 0);
    std::vector<int> keep;
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const int i = order[oi];
        if (suppressed[i]) continue;
        keep.push_back(i);
        const Prop& a = props[i];
        const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            const int j = order[oj];
            if (suppressed[j]) continue;
            const Prop& b = props[j];
            const float ix1 = std::max(a.x1, b.x1);
            const float iy1 = std::max(a.y1, b.y1);
            const float ix2 = std::min(a.x2, b.x2);
            const float iy2 = std::min(a.y2, b.y2);
            const float iw = std::max(0.f, ix2 - ix1);
            const float ih = std::max(0.f, iy2 - iy1);
            const float inter = iw * ih;
            const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
            const float uni = area_a + area_b - inter;
            if (uni > 1e-9f && inter / uni > nms_threshold_) suppressed[j] = 1;
        }
    }

    const float inv = (scale > 1e-9f) ? (1.0f / scale) : 1.0f;
    const float fw = static_cast<float>(width);
    const float fh = static_cast<float>(height);
    out.reserve(keep.size());
    for (int i : keep) {
        const Prop& p = props[i];
        FaceDetection d;
        d.score = p.score;
        d.bbox.x1 = std::min(fw, std::max(0.f, (p.x1 - pad_w) * inv));
        d.bbox.y1 = std::min(fh, std::max(0.f, (p.y1 - pad_h) * inv));
        d.bbox.x2 = std::min(fw, std::max(0.f, (p.x2 - pad_w) * inv));
        d.bbox.y2 = std::min(fh, std::max(0.f, (p.y2 - pad_h) * inv));
        if (d.bbox.x2 <= d.bbox.x1 || d.bbox.y2 <= d.bbox.y1) continue;
        // Landmarks are NOT clamped to the frame: the aligner needs the real
        // geometry, and a landmark just outside the border still describes the
        // pose correctly.
        for (int k = 0; k < 5; ++k) {
            d.landmarks[k].x = (p.kps[k * 2] - pad_w) * inv;
            d.landmarks[k].y = (p.kps[k * 2 + 1] - pad_h) * inv;
        }
        out.push_back(d);
    }
    }  // !props.empty()

    const auto t_post1 = std::chrono::steady_clock::now();
    static int s_calls = 0;
    if (++s_calls >= 30) {
        s_calls = 0;
        const int letterbox_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(t_lb1 - t_lb0).count());
        const int run_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(t_run1 - t_lb1).count());
        const int post_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(t_post1 - t_run1).count());
        MA_LOGI(TAG, "scrfd timing: letterbox=%dms run=%dms post=%dms", letterbox_ms,
                run_ms, post_ms);
    }
    return true;
}

}  // namespace face_recognition
