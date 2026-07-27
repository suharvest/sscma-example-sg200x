#include "mosaic_lut.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstring>

/* _IOWR expands differently per libc; compute the request the same way the
 * kernel header does rather than reimplementing the encoding by hand. */
#include <sys/ioctl.h>
#ifndef _IOWR
#error "_IOWR unavailable"
#endif
#define RGN_IOC_S_CTRL_U _IOWR(RGN_IOC_MAGIC, RGN_IOC_BASE + 1, struct rgn_ext_control_u)

int mosaic_lut_open(void)
{
    return ::open("/dev/cvi-rgn", O_RDWR);
}

void mosaic_lut_close(int fd)
{
    if (fd >= 0) ::close(fd);
}

static int lut_ioctl(int fd, rgn_mosaic_lut_u* lut, const uint8_t* colors)
{
    rgn_ext_control_u ctl;
    memset(&ctl, 0, sizeof(ctl));
    ctl.id = RGN_IOCTL_SDK_CTRL;
    ctl.sdk_id = RGN_SDK_SET_MOSAIC_LUT;
    ctl.handle = 0;
    ctl.ptr1 = lut;
    ctl.ptr2 = const_cast<uint8_t*>(colors);
    return ::ioctl(fd, RGN_IOC_S_CTRL_U, &ctl);
}

int mosaic_lut_query(int fd, int dev_id, int chn_id, rgn_mosaic_lut_u* out)
{
    if (fd < 0 || out == nullptr) return -1;
    memset(out, 0, sizeof(*out));
    out->dev_id = dev_id;
    out->chn_id = chn_id;
    /* Zero length and no colour pointer: the driver drops any stored table and
     * reports the layout, which is exactly what a caller needs before it can
     * size one. */
    out->lut_len = 0;
    return lut_ioctl(fd, out, nullptr);
}

int mosaic_lut_apply(int fd, int dev_id, int chn_id,
                     const uint8_t* colors, uint32_t len)
{
    if (fd < 0 || colors == nullptr || len == 0) return -1;
    rgn_mosaic_lut_u lut;
    memset(&lut, 0, sizeof(lut));
    lut.dev_id = dev_id;
    lut.chn_id = chn_id;
    lut.lut_len = len;
    return lut_ioctl(fd, &lut, colors);
}

bool mosaic_lut_supported(int fd, int dev_id, int chn_id)
{
    if (fd < 0) return false;
    rgn_mosaic_lut_u probe;
    memset(&probe, 0, sizeof(probe));
    probe.dev_id = dev_id;
    probe.chn_id = chn_id;
    probe.lut_len = 0;
    probe.grid_size = 0xFFFFu;  /* sentinel; the patched handler overwrites it */
    (void)lut_ioctl(fd, &probe, nullptr);
    return probe.grid_size != 0xFFFFu;
}
