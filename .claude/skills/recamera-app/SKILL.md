---
name: recamera-app
description: 在 reCamera 上新建或改造一个 C++ 应用时使用 —— 打包成 deb、在新版 supervisor（console）的应用画廊里上架、以及该复用哪个公共组件（隐私打码、调试串流、RTSP、MQTT/HA、ONVIF、坐标类型）。当任务涉及 solutions/ 下新增应用、修改 manifest / init 脚本 / control 脚本、应用没出现在 console 里、或要给应用加打码/推流/上报能力时触发。
---

# reCamera 应用：上架规范与公共能力

这份文档回答两个问题：**怎么让 console 认得你的应用**，以及**哪些能力已经有人写好了，别再写一遍**。

内容全部来自真实实现（`solutions/supervisor/main/src/api_app.cpp` 的校验逻辑、`cmake/package.cmake`、各组件头文件），不是设想的规范。**改动这些实现时，这份文档要跟着改**。

---

## 一、最小可上架应用

```
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

**摄像头是硬独占的。** 两个应用同时开 VPSS，轻则出流失败，重则内核挂死到要断电。

console 的模型是：**所有画廊应用永远保持 K**，跑哪个由 `/userdata/local/apps/state.json` 决定，开机时 `app_restore` 只拉起那一个。发 `S92` 意味着开机自启，会和 console 启动的那个撞车。

> 真实事故：`detection-blur` 曾经发 `S92`，是六个视觉应用里唯一一个。装上就埋了一颗定时炸弹，下次重启才炸。

脚本模板见任一现有应用，关键是 `LD_LIBRARY_PATH` 那行不能少：

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
  "description": "...",                 // 英文，卡片正文
  "description_zh": "...",
  "type": "native",                     // native | external-firmware
  "init_script": "/etc/init.d/K92my-app",  // 必须 /etc/init.d/[SK][0-9]...，无子目录
  "image": "/apps/my-app.png",
  "rtsp_url": "rtsp://{host}:8554/live0",  // {host} 会被替换
  "mqtt_topic": "recamera/my-app/results",
  "debug_ws": { "port": 8001, "video_path": "/", "results_path": "/results" },
  "privacy_blur": true,                 // 见下节；不打码就别声明
  "models": [],                         // 可切换的备选模型
  "pipeline": [ { "name": "...", "path": "...", "task": "..." } ],  // 固定流水线，仅展示
  "requires": ["gimbal"],               // 硬件依赖，白名单外的键会被丢弃并告警
  "version": "0.1.0",                   // 打包时会被 project(VERSION) 覆盖，见下节
  "author": "Seeed Studio"
}
```

后端校验（`api_app::load_manifest_file`）会因为下列原因**静默丢弃**整个 manifest，只在日志里留一行 `LOGW`：

| 拒绝原因 | 检查 |
|---|---|
| `id` 含大写/下划线/超 64 字符 | `[a-z0-9-]` |
| **文件名与 id 不一致** | `<id>.json` |
| `type` 不是 `native` / `external-firmware` | |
| `name` 为空 | |
| `init_script` 不匹配 `/etc/init.d/[SK][0-9]...` 或含 `/` | |

**应用没出现在 console 里，先看 supervisor 日志的 LOGW**，别猜。

### 3. postinst 必须把 manifest 登记进去

console 只扫两个目录：`/usr/share/supervisor/apps/`（随 supervisor 发的内置）和 `/userdata/local/apps/`（用户装的）。**应用 deb 装到的 `/usr/share/<id>/` 哪个都不是。**

```sh
APPS_DIR=/userdata/local/apps
SHARE_DIR=/usr/share/<id>
mkdir -p "$APPS_DIR"
for f in <id>.json <id>.md <id>.zh.md; do
    [ -f "$SHARE_DIR/$f" ] && cp -f "$SHARE_DIR/$f" "$APPS_DIR/$f"
done
```

> `[ -f ]` 保护会让缺失文件**静默跳过**。`facemesh-reader` 的 `postinst` 拷了一年多不存在的 `.md`，结果 console 里那一页一直是空白，没人报过。**加了就要确认文件真的在。**

### 4. 版本号只有一个真相：`project(<id> VERSION ...)`

deb 的版本来自 solution 顶层 `CMakeLists.txt` 的 `project(<id> VERSION x.y.z)` —— CPack 用它拼包名，opkg 记录它。**manifest 里的 `version` 不是第二个真相，它是一份抄件**，console 拿它渲染卡片上的版本号。

所以 `cmake/package.cmake` 在打包时会用 `PROJECT_VERSION` **覆盖** manifest 的 `version` 字段（那条 `install()` 排在 rootfs 整目录拷贝之后，同一个目标路径后写的赢）。改源 JSON 里的版本号不影响出包结果 —— **要改版本就改 `project(VERSION)`**。构建时会打印一行确认：

```
-- Manifest version pinned to 0.2.0: fitness-trainer.json
```

生成机制够不着的是**另一份抄件**：`solutions/supervisor/rootfs/usr/share/supervisor/apps/<id>.json`。它随 supervisor 发，用来在应用还没安装时也能在画廊里展示；supervisor 构建时看不到别的 solution 的 `PROJECT_VERSION`，没有任何东西会更新它。这份是真正会烂掉的，唯一的防线是检查脚本：

```bash
scripts/check-manifest-versions.sh          # 报告漂移，有漂移则 exit 1
scripts/check-manifest-versions.sh --fix    # 按 project(VERSION) 就地改正
```

它同时校验：应用自带 manifest 的版本 == `project(VERSION)`；supervisor 内置副本的版本 == `project(VERSION)`；两份都存在时内容必须**逐字节一致**（内置副本是拷贝，不是变体）。

> 2026-07-29 首次跑这个脚本，**十个应用里九个在漂**：卡片显示 0.1.0 而装上的包是 0.2.0；`face-analysis` 和 `yolo-detector` 的 manifest 版本甚至比任何存在过的包都高（1.0.0 / 1.1.0 对 0.4.0 / 0.5.0）。这个字段从来没人维护过，因为没有任何东西会在它错的时候出声。

### 5. 发布：改了版本号不等于发布了

**deb 上传到 CDN 是发布链路里唯一没有脚本兜底的一步**，也是唯一靠人记得的一步。

> 2026-07-29 的真实代价：supervisor 0.5.0 / 0.5.1 / 0.5.2 各自构建、scp 装到设备、真机验证通过——而 CDN 上始终是 0.4.1。所有从 SenseCraft App 部署的用户拿到的都是没有这三版功能的 console，**而且没有任何地方会报错**：ecosystem 的 yaml 写 0.4.1，CDN 上也确实有 0.4.1，两者完美自洽，只是一起陈旧。

所以有效的检查不是「URL 通不通」或「校验和对不对」（当时都是对的），而是**「ecosystem 发的版本，是不是我们真正在构建的版本」**。这个问题跨两个仓库，正因如此才一直没人问。

```bash
scripts/release-app.py --check                      # 所有应用：构建版本 vs ecosystem 发布版本
scripts/release-app.py <app> --publish-content      # 上传 + 回验 + 改 yaml + validate + 重生成 catalog
```

`--check` 是护栏，**bump 完版本号、发布之前各跑一次**。发布会把包下载回来比对 sha256 —— `ossutil` 说成功和字节真的取得到不是一回事。

**发布顺序不能颠倒，而且 OTA 那一步有意留在脚本外：**

```
release-app.py <app> --publish-content    上传、改 yaml、validate、catalog
git commit                                 版本号那次提交
generate_solution_manifest.py              OTA 内容，从已提交的状态打包
```

`generate_solution_manifest.py` 拒绝在 `solutions/` 有未提交路径时运行——它的 zip 从工作区打包，先发布就等于发出没进 git 的内容。而 release 刚刚改写了 device yaml，所以那个守卫每次都会拦。**这是对的，别绕过它。**

反过来也不成立：generate 不能调 release，那会去发布还没产生的内容。依赖是单向的。

solutions 仓路径用 `$SENSECRAFT_SOLUTIONS` 指定。

> 还有一层脚本管不到：**CDN 边缘缓存**。源站更新后边缘会按 TTL 挂着旧副本一段时间（实测 `X-Cache: TCP_MEM_HIT`，`Age` 持续增长，加 query 参数也击穿不了——这个 CDN 的缓存键忽略 query string）。发完不要立刻断言"用户能拿到新版"，要么等，要么用 aliyun CLI 主动刷新。

### 6. control 脚本的执行位

`cmake/package.cmake` 把 `control/{preinst,postinst,prerm,postrm}` 打进 deb。**这些文件在 git 里必须是 `+x`**，否则设备上 `opkg` 报 126。

> Docker Desktop 的挂载缓存会吃掉 `chmod`。改过权限后 `git update-index --chmod=+x`，并重启容器再打包。

### 7. main/CMakeLists.txt

```cmake
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_LIST_DIR}/*.c ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
component_register(
    COMPONENT_NAME main
    SRCS ${srcs}
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}
    PRIVATE_REQUIREDS sscma-micro sophgo rtsp_server debug_stream privacy_blur
    REQUIREDS opencv_core opencv_imgcodecs opencv_imgproc
)
```

---

## 二、公共能力：别重写这些

| 组件 | 头文件 | 提供什么 | 什么时候用 |
|---|---|---|---|
| `sscma-micro` | `sscma.h` | 模型加载、TPU 推理、`ma_img_t` | 所有需要推理的应用 |
| `sophgo` | `video.h` | 摄像头、VPSS 通道、VENC | 所有取流的应用 |
| `rtsp_server` | `rtsp_server.h` | RTSP 出流 | 要给 NVR/VLC/VMS 拉流 |
| `debug_stream` | `debug_stream.h` | H.264-over-WS 预览、结果 JSON、`/snapshot.jpg` | 要在 console 调试页出现 |
| `privacy_blur` | `privacy_blur.h` | 编码前遮挡检测目标（三后端 + 跟踪 + 热重载） | 画面里有人 |
| `geometry` | `norm_box.h` | 带语义的归一化框类型 | **任何跨模块传检测框的地方** |
| `ha_mqtt` | `ha_mqtt.h` | MQTT 发布 + Home Assistant 自动发现 | 要上报结果 |
| `onvif_meta` | `onvif_meta.h` | ONVIF 分析元数据模型与序列化 | 要给 VMS 送结构化结果 |
| `onvif_service` | `onvif_service_bringup.h` | WS-Discovery + Device/Media2 SOAP | 要被 VMS 自动发现 |

### 隐私打码（`privacy_blur`）

**设备级设置**，配置在 `/userdata/local/blur.conf`，console 设备页管开关。应用要做的只有三件：

```cpp
// 1. 视频通路起来之后再 init（RGN 需要 VPSS 通道在跑）
PrivacyBlurConfig cfg;
loadPrivacyBlurConfig(privacy_blur::PRIVACY_BLUR_CONFIG_PATH, cfg, &err);
blur->init(cfg, stream_width, stream_height);

// 2. 每帧喂检测框 —— 必须在 returnFrame() 之前，
//    像素化后端要读这一帧的像素来取色
blur->onDetection(geometry::toStream(boxes, inf_w, inf_h, stream_w, stream_h), &frame);

// 3. 存快照前在内存里遮一遍（硬件遮罩管不到快照）
privacy_blur::pixelateRgb888(frame.data, frame.width, frame.height, snap_boxes, blur->blockPx());
```

然后在 manifest 里声明 `"privacy_blur": true`，console 调试页才会显示那个快捷开关。**不打码的应用别声明**——一个拨了对画面没反应的开关，比没有开关更像故障。

`/snapshot.jpg` 是 ONVIF `GetSnapshotUri` 公布的地址。**视频打了码而快照没打，等于没打。**

### 坐标类型（`geometry`）

检测框的两个语义在这个代码库里**不统一**，且都静默出过错：

```cpp
geometry::InferBox   // 相对推理通道归一化
geometry::StreamBox  // 相对出流归一化
```

- 两者是不同类型，**互不隐式转换**；跨参照系只有 `geometry::toStream()` 一条路
- 没有聚合初始化，只能 `fromCenter()` / `fromCorner()` —— 逼你在紧挨检测器的那一行回答"中心还是角"

各应用检测器的约定：

| 应用 | 约定 |
|---|---|
| `face-analysis` | **左上角**（FaceDetector 统一归一化成角，属性分析要按角裁剪） |
| `facemesh-reader` / `yolo-detector` / `retail-vision` / `detection-blur` | 中心 |

**face-analysis 是唯一的角制，别照抄邻居。** 改检测器的归一化约定时类型不会提醒你，要手动同步这张表和 `docs/HANDOFF-privacy-blur.md` 第九节。

VPSS 把画面按比例装进每个通道，4:3 推理通道里 16:9 画面上下留边。**推理帧坐标直接喂给出流遮罩，遮罩只有该有高度的 3/4 并朝中心收缩**——边缘的人会露脸。

---

## 三、构建与部署

```bash
# clone 后一次性：SDK 里有两个版本的 libwebsockets，误取任一个都是静默 ABI 不匹配
components/libwebsockets/fetch_and_build.sh

docker exec ubuntu_dev_x86 bash -c '
export SG200X_SDK_PATH=/workspace/sg2002_recamera_emmc
export PATH=/workspace/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
cd /workspace/sscma-example-sg200x/solutions/<id>
cmake -B build -DCMAKE_BUILD_TYPE=Release . && cmake --build build -j4 && cd build && cpack'
```

各 solution 目录下有 `deploy.sh`（编译 + 停冲突服务 + 装 deb + 起服务），日常用它。

### 验证部署的是不是你以为的那个

CPack 会 strip 二进制，所以 **构建目录里的 md5 和设备上的对不上是正常的**。要比就**解包 deb 拿里面的载荷比**：

```bash
ar x <app>_<ver>_riscv64.deb && tar xf data.tar.gz
md5sum ./usr/local/bin/<id>          # 与设备上 md5sum /usr/local/bin/<id> 对照
```

判断进程身份用 `readlink /proc/<pid>/exe`，**别 grep 命令行**（会匹配到 grep 自己）。

---

## 四、上架前自查

- [ ] init 脚本是 `K92<id>`（不是 S），且 `+x`
- [ ] `control/*` 在 git 里是 `+x`
- [ ] manifest 文件名 == `id`，`id` 只含 `[a-z0-9-]`
- [ ] `postinst` 会把 `<id>.json` / `<id>.md` / `<id>.zh.md` 拷进 `/userdata/local/apps/`
- [ ] 那三个文件**真的存在**（`.md` 缺了不会报错，只会让 console 那页空白）
- [ ] 改过版本 → 改的是 `project(<id> VERSION ...)`，且 `scripts/check-manifest-versions.sh` 干净
      （有 supervisor 内置副本的应用尤其要跑，那份没有任何东西会自动更新）
- [ ] 画面里有人 → 接了 `privacy_blur`，并且 manifest 里声明了 `privacy_blur: true`
- [ ] 接了打码 → 快照路径也遮了（否则 ONVIF 抓图是原图）
- [ ] 检测框跨模块传递用了 `geometry::InferBox` / `StreamBox`
- [ ] 集成文档里写的开关、默认值、命令行参数**与代码一致**
- [ ] 发布过了 → `scripts/release-app.py --check` 干净（改了版本号却没发布，yaml 和 CDN 会一起陈旧，任何 URL/校验和检查都发现不了）
- [ ] 装完在 console 里能看到、能切换、切换后画面正常

> 最后一条不能靠"应该没问题"。**文档说谎比没文档更贵**：`face-analysis` 的集成指南曾长期写着"默认打码模糊，`--no-blur` 关闭"，而实际默认关闭、那个参数早已删除——用户照着找不到，只会得出"功能坏了"的结论。

---

## 五、相关文档

- `docs/HANDOFF-privacy-blur.md` —— 打码全线的实现、已验/未验清单、坐标语义对照表（第九节）
- `docs/onvif-implementation-spec.md` —— ONVIF 的设计与刻意不做的部分
- `docs/kernel-build.md` —— 补丁内核模块怎么编（vermagic 陷阱）
- `solutions/supervisor/README.md` —— console 侧：应用画廊、打码、ONVIF、集成页
- `README.md` —— 工具链、libwebsockets 前置步骤、补丁内核说明
