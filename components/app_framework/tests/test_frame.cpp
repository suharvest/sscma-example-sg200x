/**
 * @file test_frame.cpp
 * @brief Host tests for af::VideoFrame layout, formats and zero-copy sub_frame().
 */
#include <cstring>
#include <vector>

#include "af/frame.h"
#include "af_test.h"

using af::PixelFormat;
using af::Rect;
using af::VideoFrame;

TEST(layout_is_one_cache_line) {
    CHECK_EQ(sizeof(VideoFrame), 64u);
    CHECK_EQ(sizeof(af::Plane), 16u);
    CHECK(std::is_trivially_copyable<VideoFrame>::value);
    VideoFrame f;
    CHECK(!f.valid());
    CHECK(!f.cpu_accessible());
}

TEST(format_tables) {
    CHECK_EQ(af::plane_count(PixelFormat::RGB888), 1);
    CHECK_EQ(af::plane_count(PixelFormat::NV21), 2);
    CHECK_EQ(af::plane_count(PixelFormat::RGB888Planar), 3);
    CHECK_EQ(af::plane_count(PixelFormat::Unknown), 0);
    CHECK_EQ(af::bytes_per_pixel(PixelFormat::BGR888), 3);
    CHECK_EQ(af::bytes_per_pixel(PixelFormat::Gray8), 1);
    CHECK(std::strcmp(af::to_string(PixelFormat::NV21), "NV21") == 0);
}

TEST(packed_defaults_stride) {
    std::vector<uint8_t> buf(8 * 4 * 3);
    VideoFrame f = VideoFrame::packed(buf.data(), 8, 4, PixelFormat::RGB888);
    CHECK(f.valid());
    CHECK(f.cpu_accessible());
    CHECK_EQ(f.planes[0].stride, 24u);
    CHECK(f.data() == buf.data());
    CHECK_EQ(f.stride(), 24u);
    CHECK(f.row(2) == buf.data() + 48);
    CHECK(f.at(3, 1) == buf.data() + 24 + 9);

    // packed() refuses multi-plane formats
    CHECK(!VideoFrame::packed(buf.data(), 8, 4, PixelFormat::NV21).valid());
}

TEST(plane_bitfields_hold_large_values) {
    af::Plane p(nullptr, 0x0000FFFF12345678ull, 11520);
    CHECK_EQ(p.phy, 0x0000FFFF12345678ull);
    CHECK_EQ(p.stride, 11520u);
}

TEST(sub_frame_rgb_is_zero_copy) {
    const int W = 10, H = 6, S = 32;  // padded stride
    std::vector<uint8_t> buf(S * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) buf[y * S + x * 3] = static_cast<uint8_t>(y * 16 + x);

    VideoFrame f = VideoFrame::packed(buf.data(), W, H, PixelFormat::RGB888, S, 0x1000);
    f.seq = 42;
    f.ts_ms = 7;

    VideoFrame c = f.sub_frame(Rect{2, 3, 4, 2});
    CHECK(c.valid());
    CHECK_EQ(c.width, 4);
    CHECK_EQ(c.height, 2);
    CHECK_EQ(c.planes[0].stride, static_cast<uint64_t>(S));
    CHECK(c.planes[0].vir == buf.data() + 3 * S + 2 * 3);
    CHECK_EQ(c.planes[0].phy, 0x1000u + 3 * S + 2 * 3);
    CHECK_EQ(*c.at(0, 0), 3 * 16 + 2);
    CHECK_EQ(*c.at(3, 1), 4 * 16 + 5);
    CHECK_EQ(c.seq, 42u);
    CHECK_EQ(c.ts_ms, 7u);

    // nested crop composes
    VideoFrame cc = c.sub_frame(Rect{1, 1, 10, 10});
    CHECK_EQ(cc.width, 3);
    CHECK_EQ(cc.height, 1);
    CHECK_EQ(*cc.at(0, 0), 4 * 16 + 3);
}

TEST(sub_frame_clips_to_bounds) {
    std::vector<uint8_t> buf(16 * 16);
    VideoFrame f = VideoFrame::packed(buf.data(), 16, 16, PixelFormat::Gray8);

    VideoFrame a = f.sub_frame(Rect{-4, -2, 8, 8});
    CHECK_EQ(a.width, 4);
    CHECK_EQ(a.height, 6);
    CHECK(a.planes[0].vir == buf.data());

    VideoFrame b = f.sub_frame(Rect{12, 14, 100, 100});
    CHECK_EQ(b.width, 4);
    CHECK_EQ(b.height, 2);
    CHECK(b.planes[0].vir == buf.data() + 14 * 16 + 12);

    // no overflow on extreme values
    VideoFrame c = f.sub_frame(Rect{0x7FFFFFF0, 0, 0x7FFFFFFF, 4});
    CHECK(!c.valid());
}

TEST(sub_frame_empty_or_invalid) {
    std::vector<uint8_t> buf(16 * 16);
    VideoFrame f = VideoFrame::packed(buf.data(), 16, 16, PixelFormat::Gray8);
    CHECK(!f.sub_frame(Rect{0, 0, 0, 5}).valid());
    CHECK(!f.sub_frame(Rect{20, 0, 5, 5}).valid());
    CHECK(!f.sub_frame(Rect{3, 3, -1, 5}).valid());
    CHECK(!VideoFrame{}.sub_frame(Rect{0, 0, 4, 4}).valid());
}

TEST(sub_frame_physical_only_keeps_null_vir) {
    VideoFrame f = VideoFrame::packed(nullptr, 64, 64, PixelFormat::RGB888, 0, 0x80000000ull);
    CHECK(f.valid());
    CHECK(!f.cpu_accessible());
    VideoFrame c = f.sub_frame(Rect{8, 8, 16, 16});
    CHECK(c.planes[0].vir == nullptr);
    CHECK_EQ(c.planes[0].phy, 0x80000000ull + 8 * 192 + 8 * 3);
}

TEST(sub_frame_nv21_aligns_chroma) {
    const int W = 8, H = 8;
    std::vector<uint8_t> y(W * H), vu(W * H / 2);
    VideoFrame f;
    f.format = PixelFormat::NV21;
    f.width = W;
    f.height = H;
    f.planes[0] = af::Plane(y.data(), 0, W);
    f.planes[1] = af::Plane(vu.data(), 0, W);

    // covers x 3..6, y 3..5 -> grown to x 2..7, y 2..5
    VideoFrame c = f.sub_frame(Rect{3, 3, 4, 3});
    CHECK(c.valid());
    CHECK_EQ(c.width, 6);
    CHECK_EQ(c.height, 4);
    CHECK(c.planes[0].vir == y.data() + 2 * W + 2);
    CHECK(c.planes[1].vir == vu.data() + 1 * W + 2);

    // a 1-pixel crop grows to the enclosing 2x2 block
    VideoFrame one = f.sub_frame(Rect{5, 5, 1, 1});
    CHECK_EQ(one.width, 2);
    CHECK_EQ(one.height, 2);
    CHECK(one.planes[0].vir == y.data() + 4 * W + 4);

    // growth never leaves the frame
    VideoFrame edge = f.sub_frame(Rect{7, 7, 5, 5});
    CHECK_EQ(edge.width, 2);
    CHECK_EQ(edge.height, 2);
    CHECK(edge.planes[0].vir == y.data() + 6 * W + 6);
}

TEST(sub_frame_planar_offsets_every_plane) {
    const int W = 4, H = 4;
    std::vector<uint8_t> r(W * H), g(W * H), b(W * H);
    VideoFrame f;
    f.format = PixelFormat::RGB888Planar;
    f.width = W;
    f.height = H;
    f.planes[0] = af::Plane(r.data(), 0, W);
    f.planes[1] = af::Plane(g.data(), 0, W);
    f.planes[2] = af::Plane(b.data(), 0, W);
    CHECK(f.cpu_accessible());

    VideoFrame c = f.sub_frame(Rect{1, 2, 2, 2});
    CHECK(c.planes[0].vir == r.data() + 2 * W + 1);
    CHECK(c.planes[1].vir == g.data() + 2 * W + 1);
    CHECK(c.planes[2].vir == b.data() + 2 * W + 1);
}

AF_TEST_MAIN
