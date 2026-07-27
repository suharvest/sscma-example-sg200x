# Handoff：隐私打码 / ONVIF / lws

2026-07-27 更新。接手前先读「已验证 vs 未验证」和「踩过的坑」两节。

---

## 一、三条线的状态

| 线 | 状态 |
|---|---|
| mongoose → libwebsockets | **已完成并提交**，仓库不再含 GPL 代码 |
| ONVIF（发现 + Device/Media2 + 元数据 + console） | **已完成并提交**，7 个 app 打通 |
| 隐私打码 | **代码完成，大部分已真机验证，全部未提交** |

打码是当前重心，也是唯一有大量未提交改动的。

---

## 二、blur-probe：先读这个

`solutions/blur-probe/` —— 用**合成检测框**跑**真实视频通路**（相机 → VPSS → RGN → VENC → RTSP），
只把检测器换掉，其余是应用跑的同一套代码。

```bash
blur-probe --pattern grid                       # 四角+中心，查偏移/缩放
blur-probe --box 0.5,0.5,0.2,0.3 --box ...      # 任意框，可重复
blur-probe --pattern two                        # 两框，查框间空隙
blur-probe --raw-snapshot                       # 关掉快照遮罩，复现漏洞
```

**它把"遮罩画得对不对"和"人脸检得准不准"彻底分开了。** 用人脸验了大半天没定位到的
两个几何 bug，换成合成框第一轮就锁定。而且框是已知量，偏差可量化，不需要人出镜。

> 坐标用**流坐标系**归一化，故意不做 letterbox 转换 —— 这样"probe 下也偏"就一定是遮罩路径的问题，
> "只有应用里偏"就一定是那个应用的转换问题。两个嫌疑人分开。

配套量测脚本在 scratchpad：按「格内平坦、格间跳变」找马赛克（墙面也平，光靠平坦度会误判）。
更准的办法见下面 force_alpha 那条。

---

## 三、已修的问题（都有实测支撑）

### 1. sscma 预处理是「拉伸」，不是 letterbox —— 这是很多误判的源头

`ma_cv.cpp` 的 `rgb888_to_rgb888_planar()`：
```c
beta_w = (sw << 16) / dw;   // 横竖各自独立，不 pad
beta_h = (sh << 16) / dh;
```
所以检测框**线性对应回它运行时的那一帧**（640×480 推理帧），不是正方形帧。
**判断任何"要不要做 letterbox 补偿"之前先回到这里。** 我在这上面推理错过一轮。

### 2. `privacy_blur` 在补偿一个不存在的 letterbox（已修）

原代码 `scale_h = stream_w/stream_h`、`offset_y = (h-w)/2`，把 `target_h` 抬成了 1280：

| 请求 h | 应得 | 实测（修前） |
|---|---|---|
| 0.10 | 72 px | 128 |
| 0.25 | 180 px | 320 |
| 0.50 | 360 px | 640 |

恒为 16/9，绝对值恰好 `h_norm × stream_width`。改成直接用 `stream_w/h`、offset 归零后三档精确吻合。
**detection-blur 也受影响**（待办 #10 未复查）。

**契约**：`BlurBox` 是**相对输出流**的归一化中心坐标。调用方若检测帧形状与流不同，
**必须自己先转** —— 只有调用方知道检测跑在什么帧上，在组件里猜就是这个 bug 的来历。
face-analysis / facemesh-reader 用 `debug_stream_letterbox_to_display()` 转（与叠加框共用实现）。

### 3. 内核「整格填充」是错的（已回退）

我曾把 `rgn.c` 的遮罩表改成铺满整个网格来掩盖黑带。实测：**5 个小框 → 1 块覆盖整个外接矩形的
巨型马赛克**，框越分散误遮越大。已回退成只填各矩形内部。

> handoff 里预先写的反例判据这次直接生效 —— 现象一出来就知道退哪个 hunk，不用重新推理。
> **这类判据值得继续写。**

### 4. 黑带根因是 `force_alpha`（已修，默认改为 0）

| `mask_force_alpha` | 块间空隙 | `mask_alpha` 可调 |
|---|---|---|
| 1（原来） | **纯黑** | ✅ |
| 0（现默认） | **透明** | ✅ 仍有效 |

原以为要在"可调透明度"和"正确几何"间二选一，寄存器描述读起来是这样，**实测两者不冲突**。

> **副产品：`force_alpha=1` 是精确测量工具。** 纯黑在自然画面里不出现，两个分开的框之间那条
> 黑缝边界可以像素级定位。目测同一矩形三帧给出 1.56/1.78/1.89 三个比值，用黑缝量则四帧完全一致。

### 5. 大框守卫让人越近越不打码（已修）

`box.w > 0.7 && box.h > 0.7` 会**整条跳过遮罩**。它防的是 ISP 初始化期的噪声帧，但用"框有多大"
代理"是不是初始化噪声"，把正常大特写一并误伤 —— 最该保护的近距离场景保护最弱。
改为守卫只在预热 30 帧内生效，之后按普通方式裁剪。

### 6. 块大小固定导致保护随距离衰减（已修）

硬件块只有 8/16 两档，调不大。绕法：**我们掌握颜色表**，让 k×k 个 16px 格共用一色，
视觉即 16k 像素块，硬件仍走 16px，零额外开销。`BLUR_BLOCKS_PER_TARGET` 默认 12，0 退回固定块。
实测：框宽 1024px → k=6 → 中位连续段从 29px 变 92px。

### 7. `/snapshot.jpg` 绕过硬件遮罩（已修）

快照走 `debug_stream_offer_snapshot(rgb888)` —— 应用喂的推理帧 + OpenCV 软编，
**完全绕过 VPSS/VENC**，而它正是 ONVIF `GetSnapshotUri` 公布的地址。
新增 `privacy_blur::pixelateRgb888()`，仅在快照被请求时（`debug_stream_snapshot_armed()`）执行。

> facemesh-reader 还另有一个既有 bug：`offer_snapshot` 被关在 HA 判断的大括号内，
> **只有开了 HA 且检出有效人脸时快照才更新**。已一并移出。

### 8. 配置改动不再一律重启（已修）

原来改任何字段都重启应用。**alpha 尤其不该重启** —— 没人能透过一个每次调整都变黑的取景器调透明度。

| 字段 | 重启 | 依据 |
|---|---|---|
| `alpha` / `enabled` / `blocks_per_target` | ❌ | 内核每帧读 sysfs；`setEnabled()` 是原子量；填表时才读 |
| `backend` / `block_px` / `max_regions` | ✅ | 决定 RGN 区域分配，要重建 |

应用侧在预测线程里每秒 `stat` 一次配置文件。**用轮询不用 inotify**：supervisor 是 rename 原子替换写的，
inotify 的 watch 会跟着旧 inode 失效。supervisor 侧额外直接写一次 alpha 的 sysfs，
这样没有应用在跑也生效。实测 PID 不变、sysfs 150→140。

---

## 四、当前未解决的问题（下一步的起点）

### ⚠️ P0：移动时颜色表被拒，遮罩退化成纯色块

**现象**：人一动，遮罩变成半透明纯藏青矩形（RGB332 的 index 1），脸从底下透出来。
静止时正常显示彩色马赛克。

**机理**：
```
applyRegions:  设置区域显示属性 → query 布局 → 填表 → 上传
驱动:          下次硬件配置时才用，且只在 stride/grid_w/grid_h 全一致时才认
预测线程:      每 33ms 挪一次区域，但没有能力重填表（手里没有像素）
```
人一动，格子布局变了，上一帧的表对不上，驱动整张丢弃、退回纯色。

**这条与 `alpha < 255` 叠加后是真实的隐私失效** —— 半透明纯色盖脸，人脸可辨识。
修好之前 alpha 应保持 255。

**三条候选修法**：
1. 上传表后**再触发一次硬件配置**，让驱动在几何一致的当下取用。代价：每帧多一次 RGN 调用（需实测开销）。**倾向这条**
2. 硬件路径下不让预测线程挪区域。代价：两次检测之间遮罩不跟随，人可能从遮罩底下走出来 —— 隐私上不可接受
3. 预测线程按位移**平移表内容**。代价：颜色略滞后，但保住马赛克观感

### ⚠️ P1：快速移动产生残影（修改已部署，**未验证**）

两块遮罩并存：当前检测一块，上一位置残留一块。根因是 IoU 关联 ——
**IoU 对不重叠的框恒为 0**，检测间隔 150–300ms 时人脸位移可超过自身宽度 → 关联必然失败 →
新建轨迹，旧轨迹等 `max_miss_=15` 帧才淘汰。

已加"按尺寸归一化的中心距"兜底（`kMaxAssocDist = 1.5`，另加尺寸比 0.5–2.0 的闸门防止跨目标误配）。
**但验证时画面已被 P0 的纯色退化盖住，观察不到原始现象，所以这条改动等于没验。**

---

## 五、已验证 vs 未验证

**真机验证通过**
- 硬件马赛克出图、色块取自画面（静止场景）
- 推理耗时 60-62ms，与不打码基线持平
- 自动探测选中硬件路径，无需环境变量
- 坐标链三段验算 + 三档高度精确吻合
- 快照遮罩：A/B 对照，五框内平坦格占比 0.00 → 0.24~0.57，框外 0.06
- 大框守卫：0.8×0.8 框正常渲染（旧代码会整条跳过）
- 自适应块：16px → 96px
- alpha/enabled 热生效，PID 不变
- 驱动 `.ko` 打进 deb，状态 API 从 `not_packaged` 变可用

**未验证**
- 残影修复（被 P0 掩盖，见上）
- 调试页快捷开关的**浏览器交互**（只验了后端 API 与编译）
- 驱动部署/还原**按钮**（API 通了，按钮没点过）
- retail-vision / detection-blur / facemesh-reader 的遮罩（只有 face-analysis 真验过）

---

## 六、踩过的坑

**验证方法本身出错两次，比被测对象的 bug 更贵**

1. **TUN 伪造设备存活证据**：Clash 让 ping 全通、端口全开却无应答，我据此推出"userland 被饿死"，
   **全错**。判死活要用不经过 IP 层的证据：`ioreg` 看 USB 枚举、`ifconfig` 看链路状态、
   `ping -b <iface>` 绕开隧道。详见 memory `tun-fakes-device-liveness`。
2. **测的不是我改的那个二进制**：`/tmp` 是 tmpfs，重启即清空，init 拉起的是 `/usr/local/bin` 里的旧版；
   supervisor 还会按 pidfile 把旧版拉回来抢摄像头。**跨越管理器直接起进程，测的可能不是你以为的那个。**
   现在一律按真实部署方式装到 `/usr/local/bin`（备份在 `/userdata/fa-backup/`）。

**其他**
- 判断进程身份用 `/proc/PID/exe` 符号链接，**别 grep 命令行**（会匹配到自己，我中过两次）
- 覆盖正在运行的可执行文件，`scp`/`cp` 会 `Text file busy` 且**静默失败**——先按 exe 杀干净再传
- Docker 挂载缓存会让容器看到旧文件 → md5 对照宿主，不一致就重启容器
- `/` 是只读 ext4，不 remount 的 `cp` 会失败但脚本继续跑
- 手动跑 app 必须 `sudo`，否则开不了 `/dev/cvi-tpu0`（症状像 ION 耗尽，其实是权限）
- 上游 bug：`alpha_factor = 256` 溢出成 0（寄存器 8 位）
- 同名配置键打架两次（ONVIF、打码），根子都是「只用 API 自测发现不了」——**必须手写一份裸格式测一遍**

---

## 七、代码位置

**内核**（已提交到 fork，未提 PR）
- `suharvest/osdrv` 分支 `feat/mosaic-colour-lut`
- `suharvest/reCamera-OS` 分支 `sg200x-reCamera`
- 本地 `/Users/harvest/project/recamera/reCamera-OS/`
- **必须从 fork 编，不是从 SDK 树** —— 见 `docs/kernel-build.md`（vermagic 陷阱、只读根分区、防分叉检查）
- `scripts/check-osdrv-sync.sh` 查两棵树是否分叉

**应用侧**（全部未提交）
- `components/privacy_blur/` —— 三后端 + 硬件探测 + 按面积截断 + 自适应块 + 配置热重载
- `components/debug_stream/` —— letterbox 正/逆变换（一份实现，两个方向）
- `solutions/blur-probe/` —— 验证工具
- `solutions/{detection-blur,face-analysis,retail-vision,facemesh-reader}/`
- `solutions/supervisor/` —— blur 配置 API + 驱动部署/还原 + console（设备页卡片、调试页快捷开关、集成页分组）

配置 `/userdata/local/blur.conf`：
`BLUR_ENABLED` / `BLUR_BACKEND` / `BLUR_BLOCK_PX` / `BLUR_MAX_REGIONS` / `BLUR_ALPHA` / `BLUR_BLOCKS_PER_TARGET`

**`.ko` 不进版本库**（`rootfs/usr/share/supervisor/ko/.gitignore`），打包前先编内核。
代价：clone 后直接打包会得到按钮变灰的 deb。**已与用户确认接受。**

---

## 八、设备状态

`192.168.42.1`。内核 `cv181x_rgn.ko` `feccbf98` / `cv181x_vpss.ko` `d4f9f4b5`，
`force_alpha=0`。supervisor 0.3.0 已装。

备份：
- `/userdata/ko-backup/` 原厂驱动，md5 `3934b79e`(rgn) / `5570faa9`(vpss)
- `/userdata/fa-backup/` `supervisor.orig` `6b356b4b`、`face-analysis.orig` `e46da3d2`
- 本机 scratchpad 另有一份驱动备份

**换驱动出问题就从 `/userdata/ko-backup/` 还原 + 重启。**
