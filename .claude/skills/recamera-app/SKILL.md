---
name: recamera-app
description: 在 reCamera (SG2002 RISC-V SoC) 上新建或改造高性能 C++ 应用时使用 —— 基于 components/app_framework 框架开发（流式监控与按需触发双范式）、零拷贝切片与待机降频、打包成 deb、在新版 supervisor（console）的应用画廊里上架、以及复用公共组件（隐私打码、调试串流、RTSP、MQTT/HA、ONVIF）。当任务涉及 solutions/ 下新增应用、修改 manifest / init 脚本 / control 脚本、应用重构、工业工位质检开发、或性能优化时触发。
---

# reCamera 高性能应用开发与上架规范 (recamera-app)

面向 **ReCamera（算能 SG2002，单核 RISC-V 64-bit @ 1.0 GHz，256MB / 512MB RAM，1TOPS INT8 TPU）** 的全套架构规范、上架规则与工程踩坑宝典。

这份文档回答三个核心问题：
1. **怎么用最少代码、最低负荷开发高性能应用**（基于 `components/app_framework`）；
2. **怎么让 supervisor / console 认得并正确管理你的应用**（Gallery 上架规范）；
3. **真实真机环境下有哪些绝对不能踩的硬件与驱动红线**。

---

## 一、核心开发架构：`components/app_framework`

在过去的旧应用中，每个方案的 `main.cpp` 都在重复手写 400+ 行相机初始化（四连 ctrl）、通道分配、RTSP、WebSocket 调试流、MQTT 与信号监听。
**新应用及老方案重构一律基于 `components/app_framework` 开发。** 框架提供流畅的 C++17 链式装配语法（Fluent API），彻底吞噬脚手架代码。

### 1. 范式 A：通用流式监控应用（目标检测、人员追踪、安防）
适用于 7x24 小时连续推流、实时计算的目标检测应用（参考标杆：`solutions/yolo-detector`）。

```cpp
#include <af/app.h>
#include "detector.h"

int main(int argc, char** argv) {
    af::App app("yolo-detector", argc, argv);
    app.with_camera({640, 640, 8})         // 推理通道 CH0 (640x640 @ 8fps)
       .with_live_view(8001)               // CH2 720p@30 H.264 只走 WebSocket，不启动 RTSP
    // 需要 NVR/ONVIF 时: .with_stream_mode(af::StreamMode::Both, af::StreamOptions{}, 8001, 8554)
       .with_mqtt();                       // MQTT Discovery 上报

    Detector detector("/userdata/local/models/yolo11n_cv181x_int8.cvimodel");

    app.on_frame([&](const af::VideoFrame& frame, af::Result& result) {
        auto boxes = detector.detect(frame);
        result.set_boxes(boxes.data(), boxes.size());
        result.json = build_mqtt_payload(boxes); // 自动由 MQTT 与 Debug WS 分发
    });

    return app.run(); // 自动托管信号退出与资源优雅释放
}
```

### 2. 范式 B：工业工位纯触发质检应用（光电开关、PLC、HTTP 同步触发）
适用于工厂产线工位：平时待机极低功耗，工件到位触发单帧抓拍，跑多步早停流水线（参考标杆：`solutions/trigger-inspection-demo`）。

```cpp
#include <af/app.h>
#include <af/pipeline.h>
#include <af/trigger.h>

struct StationState {
    af::Rect workpiece_roi;
    bool is_defect = false;
    std::string barcode;
};

int main(int argc, char** argv) {
    af::App app("trigger-qa-demo", argc, argv);
    
    // 关键：配置 idle_fps=1。平时主线程挂起，相机底层以 1fps 保活供 snapshot，用户态 CPU 占用严格保持 0%
    app.with_camera({640, 480, /*active_fps=*/15, /*physical=*/false, /*idle_fps=*/1})
       .with_stream_mode(af::StreamMode::None);  // 工位默认不开任何流（无 VENC/WS/RTSP）
    // 需要控制台预览时: .with_stream_mode(af::StreamMode::WsOnly, af::StreamOptions{640, 360, 10, 0, 0}, 8001)

    // 编排多步骤流水线（0 堆分配，支持逐 Step 耗时追踪与早停）
    af::Pipeline<StationState> pipe;
    pipe.add_step("locate", [](af::Context<StationState>& c) {
        c.state.workpiece_roi = locate_workpiece(c.frame);
        if (!c.state.workpiece_roi.valid()) return c.reject("workpiece_missing"); // 早停 Fail-fast
        return af::Flow::Next;
    });
    pipe.add_step("inspect", [](af::Context<StationState>& c) {
        // 关键：零拷贝局部 ROI 切片（耗时仅 ~5 微秒，不发生任何 memcpy）
        auto roi_frame = c.frame.sub_frame(c.state.workpiece_roi);
        c.state.is_defect = check_defect(roi_frame);
        return af::Flow::Next;
    });

    // 支持 HTTP 同步响应触发 (/trigger?tag=xxx) 与硬件 GPIO 边沿中断
    af::HttpTrigger http_trig(8090, /*sync_response=*/true);
    app.with_trigger(http_trig);

    app.on_trigger([&](const af::TriggerEvent& ev, const af::VideoFrame& frame, af::Result& r) {
        StationState state;
        auto report = pipe.run(frame, state); // 单次触发总耗时仅 ~2ms
        r.attach(report);
        r.json = format_verdict_json(state, report);
    });

    return app.run(); // 待机阻塞在 epoll
}
```

---

## 二、性能优化黄金军规

1. **零拷贝切片优先（`sub_frame`）**：
   - 严禁为了截取子区域（如人脸、缺陷部位、条码）使用 `cv::Mat::clone()` 或 `malloc`。
   - 使用 `c.frame.sub_frame(rect)`，它只修改 Plane 的起始偏移与 stride，切片耗时仅 **5 微秒**。
2. **待机动态降频（`set_camera_fps`）**：
   - 待机无任务时，通过 `app.set_camera_fps(1)` 将底层采样率降至 1fps，可彻底消除 30fps 下每秒 13MB 的无谓图像拷贝，设备长期保持 35~40℃ 常温。
3. **早停保护（Fail-fast）**：
   - 多模型级联时，轻量门控放前面。若前序步骤失败，直接返回 `c.reject()`，后面的大模型不予执行，保护 SG2002 单核 1GHz 算力。
4. **延迟构建 Debug JSON**：
   - 框架内部仅在检测到前端有 WebSocket 连接时才构造序列化 JSON，无客户端时自动跳过字符串拼接。

---

## 三、最小可上架应用（Supervisor Gallery 规范）

```text
solutions/<id>/
├── CMakeLists.txt              # 引 toolchain + project.cmake
├── main/
│   ├── CMakeLists.txt          # component_register(...)
│   └── main.cpp
├── control/
│   ├── postinst                # 必须 +x，负责把 manifest 登记进 console
│   └── prerm
└── rootfs/
    ├── etc/init.d/K92<id>      # 必须 K 开头，必须 +x
    └── usr/share/<id>/
        ├── <id>.json           # manifest，文件名必须等于 id
        ├── <id>.md             # 集成文档（英文）
        └── <id>.zh.md          # 集成文档（中文）
```

### 1. init 脚本必须是 `K92<id>`，不是 `S92`
**摄像头硬件是硬独占的。** 两个应用同时开 VPSS，轻则出流失败，重则内核挂死。
console 的模型是：**所有画廊应用永远保持 K 开头**，跑哪个由 `/userdata/local/apps/state.json` 决定，开机时 `app_restore` 只拉起那一个。脚本模板关键包含：
```sh
export LD_LIBRARY_PATH=/mnt/system/lib:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd:/mnt/system/lib/3rd:/lib:/usr/lib
```

### 2. manifest：console 的唯一入口
```jsonc
{
  "id": "my-app",                       // [a-z0-9-]，1-64 字符；文件名必须是 <id>.json
  "name": "My App",
  "name_zh": "我的应用",
  "scene": "Safety",                    // 画廊分类
  "scene_zh": "作业安全",
  "description": "...",
  "description_zh": "...",
  "type": "native",                     // native | external-firmware
  "init_script": "/etc/init.d/K92my-app",
  "image": "/apps/my-app.png",
  "rtsp_url": "rtsp://{host}:8554/live0",
  "mqtt_topic": "recamera/my-app/results",
  "debug_ws": { "port": 8001, "video_path": "/", "results_path": "/results" },
  "models": [],
  "version": "0.1.0",
  "author": "Seeed Studio"
}
```

### 3. postinst 必须把 manifest 登记进去
```sh
APPS_DIR=/userdata/local/apps
SHARE_DIR=/usr/share/<id>
mkdir -p "$APPS_DIR"
for f in <id>.json <id>.md <id>.zh.md; do
    [ -f "$SHARE_DIR/$f" ] && cp -f "$SHARE_DIR/$f" "$APPS_DIR/$f"
done
```

### 4. control 脚本必须 +x
`control/{preinst,postinst,prerm,postrm}` 打进 deb，在 git 里必须是 `chmod +x`，否则设备上 `opkg` 报 126。

---

## 四、真机驱动避坑军规

1. **绝对禁止 `kill -9`**：
   强杀持有 `/dev/cvi-vpss` 的进程会导致内核驱动的 Group 硬件句柄处于占用状态未释放，后续任何应用启动均会报 `CVI_VPSS_CreateGrp failed with -1073315836`。必须通过 `kill -15`（SIGTERM）或执行 `/etc/init.d/K92xxx stop` 优雅停止。
2. **互斥停止机制**：
   在任何应用的 `init.d` 启动脚本中，启动前必须执行 `stop` 逻辑检查并停止可能互斥占有相机的服务（`yolo-detector`, `microduck-pose-qa`, `trigger-inspection-demo`, `face-analysis` 等）。
3. **流式应用绝对严禁 `physical = true`**：
   在 `af::CameraOptions` 中默认保持 `physical = false`。SG2002 单核 1GHz 推理耗时（100~300ms）远大于相机帧间隔（33~100ms）。由于 Sophgo VPSS 驱动缺少带引用计数的持久锁帧接口，开启物理地址直传会导致硬件 DMA 覆写同一块 VB 环形池，产生严重的**图像撕裂与置信度跳变**。框架层的快速局部内存拷贝耗时仅 1~2ms，是保证生命周期安全的底线。
4. **ION Carveout 内存池超限防护**：
   ReCamera 仅有 256MB 物理 RAM，NPU 专用的 ION carveout 上限约为 40MB。多模型应用（如 SCRFD + ArcFace + 活体 + FaceMesh）必须设计分级加载与优雅降级机制（Optional init），并在关键阶段读取 `/proc/cvitek/ion` 监控内存水位。
5. **USB 虚拟网卡连接**：
   主机直连 USB-RNDIS 网卡（IP `192.168.42.1`）时，SSH 必须直连（配置 `ProxyCommand=none`），设备默认账户 `recamera`，密码 `recamera.1`。

---

## 五、Docker 交叉编译命令速查

```bash
docker run --rm \
  -v /home/harvest/project/recamera:/recamera \
  sophgo/tpuc_dev:v3.1 \
  bash -c "
    export PATH=/recamera/host-tools/gcc/riscv64-linux-musl-x86_64/bin:\$PATH
    export SG200X_SDK_PATH=/recamera/sg2002_recamera_emmc
    cd /recamera/sscma-example-sg200x/solutions/<solution-name>
    rm -rf build && mkdir build && cd build
    cmake .. && make -j\$(nproc) && cpack
  "
```
