#ifndef _MOSAIC_LUT_H_
#define _MOSAIC_LUT_H_

#include <cstdint>

/*
 * Userspace side of the hardware colour mosaic.
 *
 * The CV181x privacy-mask unit already walks a byte-per-grid-cell table out of
 * DRAM and blends it over the frame; with mask_rgb332 enabled each byte *is*
 * the cell colour. The stock driver fills that table from get_random_u32(),
 * which is why the stock "mosaic" renders as television static. A patched
 * driver exposes RGN_SDK_SET_MOSAIC_LUT so the application can supply real
 * colours instead, and the compositing stays entirely in hardware.
 *
 * That is the whole point of this path: the software OVERLAY route costs a
 * 3.6 MB clear plus a 3.6 MB upload per frame (measured at +38ms), whereas
 * this uploads a few hundred bytes and the scaler does the rest.
 *
 * Declared here rather than including the kernel uapi header, because the
 * header lives in the SDK's osdrv tree which is not on the application include
 * path. The struct layout is packed and must track
 * osdrv/interdrv/include/chip/cv181x/uapi/linux/rgn_uapi.h exactly.
 */

#define RGN_IOC_MAGIC        'V'
#define RGN_IOC_BASE         0x20
#define RGN_SDK_SET_MOSAIC_LUT 12   /* enum RGN_SDK_CTRL */
#define RGN_IOCTL_SDK_CTRL   2      /* enum RNG_IOCTL */

struct rgn_ext_control_u {
    uint32_t id;
    uint32_t sdk_id;
    uint32_t handle;
    void* ptr1;
    void* ptr2;
} __attribute__((packed));

struct rgn_mosaic_lut_u {
    int32_t dev_id;
    int32_t chn_id;
    uint32_t lut_len;
    int32_t start_x;
    int32_t start_y;
    uint16_t grid_size;
    uint16_t grid_w;
    uint16_t grid_h;
    uint32_t stride;
} __attribute__((packed));

/* Open /dev/cvi-rgn. Returns -1 when the node is absent, which is the honest
 * signal that this kernel has no colour-mosaic support. */
int mosaic_lut_open(void);
void mosaic_lut_close(int fd);

/* Ask the driver what grid the currently shown MOSAIC regions produce. Must be
 * called after the regions' display attributes are set, because the layout is
 * derived from them. Returns 0 on success. */
int mosaic_lut_query(int fd, int dev_id, int chn_id, rgn_mosaic_lut_u* out);

/* Hand over one RGB332 byte per cell. `colors` must be stride*grid_h long. */
int mosaic_lut_apply(int fd, int dev_id, int chn_id,
                     const uint8_t* colors, uint32_t len);

/*
 * Whether this kernel understands RGN_SDK_SET_MOSAIC_LUT at all.
 *
 * Return codes cannot answer this: a patched kernel with no MOSAIC region
 * attached yet fails too, and a stock kernel's unknown-sdk_id path fails with
 * an errno that carries no more meaning. What does distinguish them is that the
 * patched handler always writes the grid layout back to userspace -- even on
 * failure, because that is how a caller learns the shape it must colour for.
 * So a sentinel that survives the call means nothing wrote to the struct, which
 * means the ioctl was never handled.
 */
bool mosaic_lut_supported(int fd, int dev_id, int chn_id);

#endif /* _MOSAIC_LUT_H_ */
