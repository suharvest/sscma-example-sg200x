#include "face_align.h"

#include <cmath>
#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace face_recognition {

// Identical to align.py's ARCFACE_DST and to the template face-analysis uses.
const float kArcFaceDst112[10] = {
    38.2946f, 51.6963f,  // left eye
    73.5318f, 51.5014f,  // right eye
    56.0252f, 71.7366f,  // nose tip
    41.5493f, 92.3655f,  // left mouth corner
    70.7299f, 92.2041f,  // right mouth corner
};

bool solveSimilarity2D(const float src[10], const float dst[10], double M[6]) {
    const int N = 5;
    double ata[4][4] = {{0}};
    double atb[4] = {0};
    for (int i = 0; i < N; ++i) {
        const double sx = src[i * 2], sy = src[i * 2 + 1];
        const double dx = dst[i * 2], dy = dst[i * 2 + 1];
        const double r1[4] = {sx, -sy, 1, 0};
        const double r2[4] = {sy, sx, 0, 1};
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) ata[j][k] += r1[j] * r1[k] + r2[j] * r2[k];
            atb[j] += r1[j] * dx + r2[j] * dy;
        }
    }

    // Gaussian elimination with partial pivoting.
    double aug[4][5];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) aug[i][j] = ata[i][j];
        aug[i][4] = atb[i];
    }
    for (int i = 0; i < 4; ++i) {
        int pivot = i;
        for (int k = i + 1; k < 4; ++k) {
            if (std::abs(aug[k][i]) > std::abs(aug[pivot][i])) pivot = k;
        }
        if (std::abs(aug[pivot][i]) < 1e-9) return false;
        if (pivot != i) {
            for (int j = 0; j < 5; ++j) std::swap(aug[i][j], aug[pivot][j]);
        }
        for (int k = i + 1; k < 4; ++k) {
            const double f = aug[k][i] / aug[i][i];
            for (int j = i; j < 5; ++j) aug[k][j] -= f * aug[i][j];
        }
    }
    double x[4];
    for (int i = 3; i >= 0; --i) {
        double sum = aug[i][4];
        for (int j = i + 1; j < 4; ++j) sum -= aug[i][j] * x[j];
        x[i] = sum / aug[i][i];
    }
    for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(x[i])) return false;
    }

    const double a = x[0], b = x[1], tx = x[2], ty = x[3];
    M[0] = a;   M[1] = -b;  M[2] = tx;
    M[3] = b;   M[4] = a;   M[5] = ty;
    return true;
}

bool alignArcFace112(const uint8_t* rgb, int width, int height,
                     const Landmarks5& landmarks,
                     std::array<uint8_t, 112 * 112 * 3>& aligned_rgb) {
    if (!rgb || width <= 0 || height <= 0) return false;

    float src[10];
    for (int i = 0; i < 5; ++i) {
        if (!std::isfinite(landmarks[i].x) || !std::isfinite(landmarks[i].y)) return false;
        src[i * 2] = landmarks[i].x;
        src[i * 2 + 1] = landmarks[i].y;
    }

    double m[6];
    if (!solveSimilarity2D(src, kArcFaceDst112, m)) return false;

    cv::Mat M(2, 3, CV_64F);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) M.at<double>(r, c) = m[r * 3 + c];
    }

    const cv::Mat frame(height, width, CV_8UC3, const_cast<uint8_t*>(rgb));
    cv::Mat dst(112, 112, CV_8UC3, aligned_rgb.data());
    cv::warpAffine(frame, dst, M, cv::Size(112, 112), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return dst.data == aligned_rgb.data();
}

}  // namespace face_recognition
