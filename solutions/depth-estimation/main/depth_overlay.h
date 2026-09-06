#ifndef _DEPTH_OVERLAY_H_
#define _DEPTH_OVERLAY_H_

#include <cstdint>
#include <vector>

namespace depth {

/*
 * Depth preview as a picture-in-picture tile on the encoded stream.
 *
 * A tile, not a full-frame tint. Colouring all of 1280x720 means uploading a
 * 3,686,400-byte ARGB canvas row by row on every inference frame; the cost of
 * that path is written up in components/privacy_blur/src/privacy_blur.cpp
 * (696-702, 847-849, 909-923). 320x180 is 1/32 of the pixels and leaves the
 * actual picture visible, which is what an operator is looking at the stream
 * for.
 *
 * One OVERLAY_RGN region, not OVERLAYEX: rgn.c:1371-1376 refuses to create an
 * OVERLAYEX region unless VPSS runs in RGNEX mode, and that mode reserves ION
 * up front (see the note at privacy_blur.cpp:685). Plain OVERLAY has no such
 * precondition.
 *
 * Ordinary overlay regions on one VPSS channel may not intersect, whatever
 * layer they sit on (rgn.c:340-360), so this tile and a privacy mask cannot
 * coexist here. The application declares "privacy_blur": false and never
 * instantiates one.
 */
class DepthOverlay {
public:
    DepthOverlay() = default;
    ~DepthOverlay();

    DepthOverlay(const DepthOverlay&)            = delete;
    DepthOverlay& operator=(const DepthOverlay&) = delete;

    /* Must be called AFTER Camera::startStream(): the VPSS channel has to
     * exist before a region can attach to it. Returns false and leaves the
     * application running without a preview if the region cannot be created. */
    bool init(int stream_w, int stream_h, int pip_w, int pip_h, int margin,
              int vpss_grp, int vpss_chn);
    void deinit();

    bool ready() const { return ready_; }

    /* Colour `depth` (row-major, dw x dh) into the tile and push it to the
     * region. p02/p98 are the frame's stabilised range; smaller depth is
     * nearer and is drawn red, larger is drawn blue. */
    void update(const std::vector<float>& depth, int dw, int dh, float p02, float p98);

private:
    std::vector<uint32_t> canvas_;
    /* Column map from the depth map onto the tile; rebuilt only if either
     * width changes. */
    std::vector<int> xmap_;
    int xmap_src_w_ = -1;
    int pip_w_    = 0;
    int pip_h_    = 0;
    int vpss_grp_ = 0;
    int vpss_chn_ = 0;
    int handle_   = -1;
    bool ready_   = false;
    bool shown_   = false;
};

}  // namespace depth

#endif  // _DEPTH_OVERLAY_H_
