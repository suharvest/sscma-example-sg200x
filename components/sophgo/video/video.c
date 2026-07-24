#include "video.h"

#include "app_ipcam_camera_conf.h"
#include "app_ipcam_fv_monitor.h"

static bool is_started   = false;
static bool video_mirror = false;
static bool video_flip   = false;

static int setVbPool(video_ch_index_t ch, const video_ch_param_t* param) {
    APP_PARAM_SYS_CFG_S* sys = app_ipcam_Sys_Param_Get();

    if (ch >= sys->vb_pool_num) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "ch(%d) > vb_pool_num(%d)\n", ch, sys->vb_pool_num);
        return -1;
    }
    if (param == NULL) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "param is null\n");
        return -1;
    }

    APP_PARAM_VB_CFG_S* vb = &sys->vb_pool[ch];
    vb->bEnable            = 1;
    vb->width              = param->width;
    vb->height             = param->height;
    vb->fmt                = (param->format == VIDEO_FORMAT_RGB888) ? PIXEL_FORMAT_RGB_888 : PIXEL_FORMAT_NV21;

    return 0;
}

static int setGrpChn(int grp, video_ch_index_t ch, const video_ch_param_t* param) {
    APP_PARAM_VPSS_CFG_T* vpss = app_ipcam_Vpss_Param_Get();

    if (grp >= vpss->u32GrpCnt) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "grp(%d) > u32GrpCnt(%d)\n", grp, vpss->u32GrpCnt);
        return -1;
    }
    if (ch >= VIDEO_CH_MAX) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "ch(%d) > VIDEO_CH_MAX(%d)\n", ch, VIDEO_CH_MAX);
        return -1;
    }
    if (param == NULL) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "param is null\n");
        return -1;
    }

    APP_VPSS_GRP_CFG_T* pgrp  = &vpss->astVpssGrpCfg[grp];
    pgrp->abChnEnable[ch]     = 1;
    pgrp->aAttachEn[ch]       = 1;
    VPSS_CHN_ATTR_S* vpss_chn = &pgrp->astVpssChnAttr[ch];
    vpss_chn->u32Width        = param->width;
    vpss_chn->u32Height       = param->height;
    vpss_chn->enPixelFormat   = (param->format == VIDEO_FORMAT_RGB888) ? PIXEL_FORMAT_RGB_888 : PIXEL_FORMAT_NV21;

    return 0;
}

static int setVencChn(video_ch_index_t ch, const video_ch_param_t* param) {
    APP_PARAM_VENC_CTX_S* venc = app_ipcam_Venc_Param_Get();

    if (ch >= venc->s32VencChnCnt) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "ch(%d) > u32ChnCnt(%d)\n", ch, venc->s32VencChnCnt);
        return -1;
    }

    if (param == NULL) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "param is null\n");
        return -1;
    }

    PAYLOAD_TYPE_E enType = PT_JPEG;
    if (VIDEO_FORMAT_H264 == param->format) {
        enType = PT_H264;
    } else if (VIDEO_FORMAT_H265 == param->format) {
        enType = PT_H265;
    }
    app_ipcam_Param_setVencChnType(ch, enType);
    APP_VENC_CHN_CFG_S* pvchn = &venc->astVencChnCfg[ch];
    pvchn->bEnable            = 1;
    pvchn->u32Width           = param->width;
    pvchn->u32Height          = param->height;
    pvchn->u32DstFrameRate    = param->fps;

    if ((VIDEO_FORMAT_RGB888 == param->format) || (VIDEO_FORMAT_NV21 == param->format)) {
        pvchn->no_need_venc = 1;
    }

    return 0;
}

int initVideo(void) {
    APP_CHK_RET(app_ipcam_Param_Load(), "load global parameter");
    video_mirror = false;
    video_flip = false;

    return 0;
}

/* Apply device-level orientation from /userdata/local/camera.conf.
 * CAM_ROTATION=180 is realized as mirror+flip; the effective values are
 * XOR-composed onto the current mirror/flip flags so an application's own
 * setVideoMirror()/setVideoFlip() calls still take effect on top of the
 * device config (and vice versa). Must run before app_ipcam_Vi_Init(),
 * which pushes the flags to VI via CVI_VI_SetChnFlipMirror. */
static void applyCameraConf(void) {
    camera_conf_t conf;
    app_ipcam_CameraConf_Load(CAMERA_CONF_PATH, &conf);

    bool rot180     = (conf.rotation == 180);
    bool mirror_eff = ((conf.mirror != 0) != rot180); /* CAM_MIRROR XOR rot180 */
    bool flip_eff   = ((conf.flip != 0) != rot180);   /* CAM_FLIP XOR rot180 */

    setVideoMirror((getVideoMirror() != 0) != mirror_eff); /* current XOR effective */
    setVideoFlip((getVideoFlip() != 0) != flip_eff);

    APP_PROF_LOG_PRINT(LEVEL_INFO, "camera.conf: mirror=%d flip=%d rotation=%d -> effective mirror=%d flip=%d\n", conf.mirror, conf.flip, conf.rotation, getVideoMirror(), getVideoFlip());
}

int deinitVideo(void) {
    app_ipcam_FvMonitor_Stop();
    if (is_started) {
        /* Wedge-defense (A): the teardown chain VENC->VPSS->VI->SYS must run to
         * completion even if an earlier step errors. The old APP_CHK_RET
         * early-returned on the first failure, so a VENC or VPSS hiccup skipped
         * VPSS/VI/SYS DeInit -> Grp(0) and VI resources leak and the next app
         * gets "Grp(0) is occupied" / a corrupted pipeline. Run every step,
         * keep the first error, and only clear is_started at the end. */
        int rc, first = 0;
        rc = app_ipcam_Venc_Stop(APP_VENC_ALL);
        if (rc != CVI_SUCCESS) { APP_PROF_LOG_PRINT(LEVEL_ERROR, "Venc Stop failed %#x\n", rc); if (!first) first = rc; }
        rc = app_ipcam_Vpss_DeInit();
        if (rc != CVI_SUCCESS) { APP_PROF_LOG_PRINT(LEVEL_ERROR, "Vpss DeInit failed %#x\n", rc); if (!first) first = rc; }
        rc = app_ipcam_Vi_DeInit();
        if (rc != CVI_SUCCESS) { APP_PROF_LOG_PRINT(LEVEL_ERROR, "Vi DeInit failed %#x\n", rc); if (!first) first = rc; }
        rc = app_ipcam_Sys_DeInit();
        if (rc != CVI_SUCCESS) { APP_PROF_LOG_PRINT(LEVEL_ERROR, "System DeInit failed %#x\n", rc); if (!first) first = rc; }
        is_started = false;
        return first;
    }
    return 0;
}

int startVideo() {
    /* Guard against double initialization - causes VPSS crash */
    if (is_started) {
        APP_PROF_LOG_PRINT(LEVEL_WARN, "startVideo() already called, skipping\n");
        return 0;
    }

    int ret = 0;

    /* device-level orientation config, must precede VI init */
    applyCameraConf();

    /* init modules include <Peripheral; Sys; VI; VB; OSD; Venc; AI; Audio; etc.> */
    ret = app_ipcam_Sys_Init();
    if (ret != 0) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "init system failed with 0x%x\n", ret);
        return ret;
    }

    ret = app_ipcam_Vi_Init();
    if (ret != 0) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "init vi module failed with 0x%x\n", ret);
        app_ipcam_Sys_DeInit();
        return ret;
    }

    ret = app_ipcam_Vpss_Init();
    if (ret != 0) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "init vpss module failed with 0x%x\n", ret);
        app_ipcam_Vi_DeInit();
        app_ipcam_Sys_DeInit();
        return ret;
    }

    ret = app_ipcam_Venc_Init(APP_VENC_ALL);
    if (ret != 0) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "init venc failed with 0x%x\n", ret);
        app_ipcam_Vpss_DeInit();
        app_ipcam_Vi_DeInit();
        app_ipcam_Sys_DeInit();
        return ret;
    }

    /* start video encode */
    ret = app_ipcam_Venc_Start(APP_VENC_ALL);
    if (ret != 0) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "start venc failed with 0x%x\n", ret);
        app_ipcam_Venc_Stop(APP_VENC_ALL);
        app_ipcam_Vpss_DeInit();
        app_ipcam_Vi_DeInit();
        app_ipcam_Sys_DeInit();
        return ret;
    }

    is_started = true;

    /* ISP is running now: start the background focus-value publisher
     * (read-only AF statistics, safe during streaming). */
    {
        APP_PARAM_VI_CTX_S* vi_ctx = app_ipcam_Vi_Param_Get();
        int vi_pipe                = (vi_ctx->u32WorkSnsCnt > 0) ? vi_ctx->astChnInfo[0].s32ChnId : 0;
        app_ipcam_FvMonitor_Start(vi_pipe);
    }

    return 0;
}

int setupVideo(video_ch_index_t ch, const video_ch_param_t* param) {
    if (ch >= VIDEO_CH_MAX) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "video ch(%d) index is out of range\n", ch);
        return -1;
    }
    if (param == NULL) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "video ch(%d) param is null\n", ch);
        return -1;
    }
    if (param->format >= VIDEO_FORMAT_COUNT) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "video ch(%d) format(%d) is not support\n", ch, param->format);
        return -1;
    }

    setVbPool(ch, param);
    setGrpChn(0, ch, param);
    setVencChn(ch, param);

    return 0;
}

int registerVideoFrameHandler(video_ch_index_t ch, int index, pfpDataConsumes handler, void* pUserData) {
    app_ipcam_Venc_Consumes_Set(ch, index, handler, pUserData);
    return 0;
}

int requestVideoIDR(video_ch_index_t ch) {
    if (ch >= VIDEO_CH_MAX) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "video ch(%d) index is out of range\n", ch);
        return -1;
    }
    return app_ipcam_Venc_RequestIDR(ch);
}

int setVideoMirror(bool mirror) {
    video_mirror = mirror;
}
int setVideoFlip(bool flip) {
    video_flip = flip;
}
int getVideoMirror() {
    return video_mirror;
}
int getVideoFlip() {
    return video_flip;
}
