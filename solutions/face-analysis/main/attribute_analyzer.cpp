#include "attribute_analyzer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define TAG "AttributeAnalyzer"

// Least-squares similarity transform (rotation + scale + translation, no shear/shear).
// Solves: [a -b tx; b a ty] * (x,y,1)^T = (x',y')^T from N >= 2 correspondences.
// Replaces cv::estimateAffinePartial2D which needs calib3d (not in reCamera SDK OpenCV 4.5).
static bool solveSimilarity2D(const float src[10], const float dst[10], cv::Mat& M_out) {
    const int N = 5;
    double ata[4][4] = {0};
    double atb[4] = {0};
    for (int i = 0; i < N; i++) {
        double sx = src[i * 2], sy = src[i * 2 + 1];
        double dx = dst[i * 2], dy = dst[i * 2 + 1];
        double r1[4] = {sx, -sy, 1, 0};
        double r2[4] = {sy, sx, 0, 1};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                ata[j][k] += r1[j] * r1[k] + r2[j] * r2[k];
            }
            atb[j] += r1[j] * dx + r2[j] * dy;
        }
    }
    // Solve 4x4 linear system via Gaussian elimination with partial pivoting
    double aug[4][5];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            aug[i][j] = ata[i][j];
        aug[i][4] = atb[i];
    }
    for (int i = 0; i < 4; i++) {
        int pivot = i;
        for (int k = i + 1; k < 4; k++)
            if (std::abs(aug[k][i]) > std::abs(aug[pivot][i]))
                pivot = k;
        if (std::abs(aug[pivot][i]) < 1e-9)
            return false;
        if (pivot != i)
            std::swap(aug[i], aug[pivot]);
        for (int k = i + 1; k < 4; k++) {
            double f = aug[k][i] / aug[i][i];
            for (int j = i; j < 5; j++)
                aug[k][j] -= f * aug[i][j];
        }
    }
    double x[4];
    for (int i = 3; i >= 0; i--) {
        double sum = aug[i][4];
        for (int j = i + 1; j < 4; j++)
            sum -= aug[i][j] * x[j];
        x[i] = sum / aug[i][i];
    }
    double a = x[0], b = x[1], tx = x[2], ty = x[3];
    M_out = cv::Mat(2, 3, CV_64F);
    M_out.at<double>(0, 0) = a;
    M_out.at<double>(0, 1) = -b;
    M_out.at<double>(0, 2) = tx;
    M_out.at<double>(1, 0) = b;
    M_out.at<double>(1, 1) = a;
    M_out.at<double>(1, 2) = ty;
    return true;
}

namespace face_analysis {

bool AttributeAnalyzer::init(const std::string& genderage_model,
                              const std::string& emotion_model,
                              const std::string& landmark_model) {
    // Initialize AGR runner
    if (!genderage_model.empty()) {
        MA_LOGI(TAG, "Loading AGR model: %s", genderage_model.c_str());

        // InsightFace genderage normalization: (pixel - 127.5) / 127.5 → input in [-1, 1].
        // Previously used ImageNet mean/std, which kept the input distribution off and
        // collapsed the continuous age regression head to its bias (always ~35 years).
        agr_runner_.setPreprocess(127.5f, 127.5f, 127.5f,
                                  1.0f / 127.5f,
                                  1.0f / 127.5f,
                                  1.0f / 127.5f);

        if (!agr_runner_.init(genderage_model)) {
            MA_LOGE(TAG, "Failed to init AGR runner");
            return false;
        }

        MA_LOGI(TAG, "AGR model loaded, input size: %d", agr_runner_.inputSize());
        genderage_ready_ = true;
    }

    // Initialize Emotion runner
    if (!emotion_model.empty()) {
        MA_LOGI(TAG, "Loading Emotion model: %s", emotion_model.c_str());

        // HSEmotion enet_b0_8: ImageNet normalization on RGB
        // mean=(0.485, 0.456, 0.406)*255, scale=1/(std*255)
        emotion_runner_.setPreprocess(123.675f, 116.28f, 103.53f,
                                      1.0f / (0.229f * 255.0f),
                                      1.0f / (0.224f * 255.0f),
                                      1.0f / (0.225f * 255.0f));

        if (!emotion_runner_.init(emotion_model)) {
            MA_LOGE(TAG, "Failed to init Emotion runner");
        } else {
            MA_LOGI(TAG, "Emotion model loaded, input size: %d", emotion_runner_.inputSize());
            emotion_ready_ = true;
        }
    }

    // Initialize Landmark runner (PFLD 5-point)
    if (!landmark_model.empty()) {
        MA_LOGI(TAG, "Loading Landmark model: %s", landmark_model.c_str());

        // PFLD: mean=127.5, scale=1/127.5 -> [-1,1]
        landmark_runner_.setPreprocess(127.5f, 127.5f, 127.5f,
                                      1.0f / 127.5f,
                                      1.0f / 127.5f,
                                      1.0f / 127.5f);
        landmark_runner_.setCropScale(1.5f);

        if (!landmark_runner_.init(landmark_model)) {
            MA_LOGE(TAG, "Failed to init Landmark runner");
        } else {
            MA_LOGI(TAG, "Landmark model loaded, input size: %d", landmark_runner_.inputSize());
            landmark_ready_ = true;
        }
    }

    return genderage_ready_;
}

std::vector<AnalyzedFace> AttributeAnalyzer::analyzeAll(
    ma_img_t* full_frame,
    const std::vector<FaceInfo>& faces) {

    std::vector<AnalyzedFace> results;
    results.reserve(faces.size());

    if (!full_frame || !full_frame->data || full_frame->width <= 0 || full_frame->height <= 0) {
        return results;
    }

    const uint8_t* frame_ptr = static_cast<const uint8_t*>(full_frame->data);
    const int fw = full_frame->width;
    const int fh = full_frame->height;

    for (const auto& face : faces) {
        AnalyzedFace analyzed;
        analyzed.face = face;

        // Convert normalized bbox to pixel coordinates
        float x1 = face.x * fw;
        float y1 = face.y * fh;
        float x2 = (face.x + face.w) * fw;
        float y2 = (face.y + face.h) * fh;

        // AGR inference
        if (genderage_ready_) {
            AgeGenderRaceResult agr;
            bool agr_ok = false;

            // Try landmark-based alignment first if landmark model is loaded
            if (landmark_ready_) {
                LandmarkResult lm;
                if (landmark_runner_.infer(frame_ptr, fw, fh, x1, y1, x2, y2, lm) && lm.ok) {
                    // ArcFace canonical 5-point template for 112x112, scaled to 96x96
                    // Original ArcFace dst_pts (112x112): left_eye=(38.2946,51.6963), right_eye=(73.5318,51.5014),
                    //   nose=(56.0252,71.7366), left_mouth=(41.5493,92.3655), right_mouth=(70.7299,92.2041)
                    const float s = 96.0f / 112.0f;
                    const float arcface_dst[10] = {
                        38.2946f * s, 51.6963f * s,  // left_eye
                        73.5318f * s, 51.5014f * s,  // right_eye
                        56.0252f * s, 71.7366f * s,  // nose_tip
                        41.5493f * s, 92.3655f * s,  // left_mouth
                        70.7299f * s, 92.2041f * s   // right_mouth
                    };

                    // Build src points from detected landmarks
                    std::vector<cv::Point2f> src_pts, dst_pts;
                    for (int i = 0; i < 5; ++i) {
                        src_pts.emplace_back(lm.pts[i * 2], lm.pts[i * 2 + 1]);
                        dst_pts.emplace_back(arcface_dst[i * 2], arcface_dst[i * 2 + 1]);
                    }

                    // Estimate similarity transform (replaces cv::estimateAffinePartial2D
                    // which needs calib3d -- not in reCamera's SDK OpenCV 4.5)
                    float src_arr[10], dst_arr[10];
                    for (int i = 0; i < 5; i++) {
                        src_arr[i * 2] = src_pts[i].x;
                        src_arr[i * 2 + 1] = src_pts[i].y;
                        dst_arr[i * 2] = dst_pts[i].x;
                        dst_arr[i * 2 + 1] = dst_pts[i].y;
                    }
                    cv::Mat M;
                    bool ok = solveSimilarity2D(src_arr, dst_arr, M);
                    if (ok) {
                        // Warp aligned face from full frame
                        cv::Mat frame_mat(fh, fw, CV_8UC3, const_cast<uint8_t*>(frame_ptr));
                        cv::Mat aligned(96, 96, CV_8UC3);
                        cv::warpAffine(frame_mat, aligned, M, cv::Size(96, 96),
                                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

                        // Run AGR on aligned 96x96 RGB
                        agr_ok = agr_runner_.inferOnAlignedRgb(aligned.data, agr) && agr.ok;
                    }
                }
            }

            // Fallback to bbox-based crop if landmark alignment failed or not loaded
            if (!agr_ok) {
                agr_ok = agr_runner_.infer(frame_ptr, fw, fh, x1, y1, x2, y2, agr) && agr.ok;
            }

            if (agr_ok) {
                analyzed.attributes.is_fairface = agr.is_fairface;
                analyzed.attributes.gender = (agr.gender == 1) ? "male" : "female";
                analyzed.attributes.gender_confidence = agr.gender_score;

                if (agr.is_fairface) {
                    // FairFace: age bins + race
                    analyzed.attributes.age_bin = agr.age;
                    analyzed.attributes.age_label = getAgeBinLabel(agr.age);
                    analyzed.attributes.age_confidence = agr.age_score;
                    analyzed.attributes.race_bin = agr.race;
                    analyzed.attributes.race_label = getRaceLabel(agr.race);
                    analyzed.attributes.race_confidence = agr.race_score;
                } else {
                    // InsightFace: continuous age in years, no race
                    analyzed.attributes.age_continuous = agr.age;
                    analyzed.attributes.age_label = std::to_string(agr.age);
                    analyzed.attributes.age_confidence = agr.age_score;
                }
            }
        }

        // Emotion inference (rate-limited via emotion_interval_)
        if (emotion_ready_) {
            const bool run_now = (frame_counter_ % emotion_interval_) == 0;
            EmotionResult emo;
            bool ok = false;

            if (run_now) {
                ok = emotion_runner_.infer(frame_ptr, fw, fh, x1, y1, x2, y2, emo) && emo.ok;
            } else {
                // Reuse cached emotion from a recent frame: best IoU match against last bbox set
                float best_iou = 0.f;
                const EmotionCache* best = nullptr;
                const float ax1 = x1, ay1 = y1, ax2 = x2, ay2 = y2;
                const float aa = std::max(0.f, ax2 - ax1) * std::max(0.f, ay2 - ay1);
                for (const auto& c : last_emotion_) {
                    const float ix1 = std::max(ax1, c.x1);
                    const float iy1 = std::max(ay1, c.y1);
                    const float ix2 = std::min(ax2, c.x2);
                    const float iy2 = std::min(ay2, c.y2);
                    const float iw = std::max(0.f, ix2 - ix1);
                    const float ih = std::max(0.f, iy2 - iy1);
                    const float inter = iw * ih;
                    const float bb = std::max(0.f, c.x2 - c.x1) * std::max(0.f, c.y2 - c.y1);
                    const float uni = aa + bb - inter;
                    const float iou = uni > 1e-6f ? inter / uni : 0.f;
                    if (iou > best_iou) { best_iou = iou; best = &c; }
                }
                if (best && best_iou > 0.3f) {
                    emo.ok = true;
                    emo.emotion = static_cast<int>(best->emotion);
                    emo.score = best->confidence;
                    ok = true;
                }
            }

            if (ok) {
                analyzed.attributes.emotion = static_cast<Emotion>(emo.emotion);
                analyzed.attributes.emotion_confidence = emo.score;
                analyzed.attributes.emotion_probs.fill(0.f);
                if (emo.emotion >= 0 && emo.emotion < 8) {
                    analyzed.attributes.emotion_probs[emo.emotion] = emo.score;
                }
            }
        }

        results.push_back(analyzed);
    }

    // Update cache from this frame's inferences (only when we actually ran emotion)
    if (emotion_ready_ && (frame_counter_ % emotion_interval_) == 0) {
        last_emotion_.clear();
        last_emotion_.reserve(results.size());
        for (const auto& r : results) {
            if (r.attributes.emotion_confidence > 0.f) {
                last_emotion_.push_back({
                    r.face.x * fw, r.face.y * fh,
                    (r.face.x + r.face.w) * fw, (r.face.y + r.face.h) * fh,
                    r.attributes.emotion, r.attributes.emotion_confidence
                });
            }
        }
    }

    ++frame_counter_;
    return results;
}

}  // namespace face_analysis
