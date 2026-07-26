# Handoff：隐私打码 / ONVIF / lws

2026-07-26。接手前先读这一页，尤其「已验证 vs 未验证」和「踩过的坑」。

---

## 一、现在处于什么状态

三条线并行推进，进度不一样：

| 线 | 状态 |
|---|---|
| mongoose → libwebsockets 迁移 | **已完成并提交**，仓库不再含 GPL 代码 |
| ONVIF（发现 + Device/Media2 + 元数据 + console 开关） | **已完成并提交**，7 个 app 全线打通 |
| 隐私打码（软件 + 硬件马赛克） | **代码完成，部分验证，全部未提交** |

打码这条线是当前重心，也是唯一有大量未提交改动的。

---

## 二、打码这件事的来龙去脉

出发点：用户要求「不能是电视机雪花，也不能还看得清人脸」。

CV181x 的 `MOSAIC` 区域类型看起来正合适，但**原厂驱动往硬件遮罩表里填的是 `get_random_u32()`**，
所以它渲染出来是黑白噪点 —— 硬件没错，喂的数据错了。

试过三条路，结论：

| 方案 | 观感 | 每帧开销 | 结论 |
|---|---|---|---|
| MOSAIC（原厂） | 黑白噪点 | 0 | ❌ 不合格 |
| COVEREX | 纯色实心块 | 0 | 满足隐私但像打码条，最多 4 个区域 |
| OVERLAY 软件合成 | 真像素化 | **+38ms、3.6MB** | 可用，但代价大 |
| **MOSAIC + 颜色 LUT** | 真像素化 | **≈0、几百字节** | ✅ 最终方案，需内核补丁 |

最后一条是把硬件本来就有的能力用起来：遮罩单元按网格读「每格一字节」的表，
`mask_rgb332=1` 时每个字节就是 RGB332 颜色，硬件自己合成。我们只需要把随机数换成真实平均色。

**应用不需要知道内核有没有补丁** —— 启动时探测，能用硬件就用，不能用静默回退软件路径，画面一样。

---

## 三、代码在哪

### 内核（已提交）

- `suharvest/osdrv` 分支 `feat/mosaic-colour-lut`，commit `7f90be2`
- `suharvest/reCamera-OS` 分支 `sg200x-reCamera`，commit `fb5b3a1`（`.gitmodules` 指向上面那个）
- 本地 `/Users/harvest/project/recamera/reCamera-OS/`

改了 4 个文件：新增 `RGN_SDK_SET_MOSAIC_LUT` ioctl、遮罩表从常量改为读用户表并回传网格布局、
`mask_rgb332=1`、`mask_alpha` 模块参数、整格填充修复。

**构建与部署流程见 `docs/kernel-build.md`**（vermagic 陷阱、只读根分区、防分叉检查）。

### 应用侧（未提交）

- `components/privacy_blur/` —— 三后端 + 硬件能力自动探测 + 按面积截断
- `solutions/{detection-blur,face-analysis,retail-vision}/` —— 接线
- `solutions/supervisor/` —— blur 配置 API + 驱动部署/还原 API + console 两处界面

配置统一在 `/userdata/local/blur.conf`：
`BLUR_ENABLED` / `BLUR_BACKEND` / `BLUR_BLOCK_PX` / `BLUR_MAX_REGIONS` / `BLUR_ALPHA`

---

## 四、已验证 vs 未验证（最重要的一节）

### 真机验证通过

- 硬件马赛克出图，色块取自画面（RTSP 实截）
- 推理耗时 **60-62ms**，与不打码基线持平（软件路径是 98ms）
- 自动探测：无任何环境变量，只凭 `blur.conf` 就选中硬件路径
- console API → `blur.conf` → 应用读取 → 画面变化，端到端闭环
- blur 配置 API：默认值、越界拒绝、落盘格式、回读一致、**裸格式兼容**
- 中性色不再偏色（遮罩主色从藏青 (0,0,32) 变成中性灰 (32,32,32)）

### 编过但没在设备上跑过

- `mask_alpha` 写 sysfs 是否立即生效
- **整格填充修复**（黑带）—— 见下面的反例
- 驱动部署/还原按钮（且 `.ko` 还没打进 deb，见待办 7）
- console 卡片在浏览器里的实际交互（只验了后端 API）

### 需要有人站在镜头前才能验

- **face-analysis 人脸遮罩落点** —— 这里修过一个坐标系 bug（`FaceInfo` 是左上角、
  `BlurBox` 是中心，直接赋值会偏半个框）。代码审过是对的，但没有人脸就证明不了
- retail-vision 打码从未跑过
- 超过 8 个目标时按面积截断

---

## 五、踩过的坑（重复踩会很贵）

**根因判断错过一次，值得记。** 黑带我判断是「格子映射出帧被跳过」，被证伪了 ——
离线模拟跑 250 万个格子证明那种情况**结构上不可能**。真因是 `force_alpha=1` 让
`no_mask_idx` 失效，包围盒内每个格子都被不透明合成，而驱动只填各区域矩形内部的格子，
**区域之间的格子从来不归任何人填**，留在索引 0 上被画成黑色。

> ⚠️ **这个修复带一个反例**：如果 `no_mask_idx` 在 `force_alpha` 下其实仍然有效，
> 那今天的空隙是透明的而非黑的，铺满整格反而会新增遮挡 —— 表现为两张脸之间多出一块
> 粗糙矩形。**看到这个现象就回退 `rgn.c` 那一处 hunk。**

**同名配置键打架，两次。**
- ONVIF：supervisor 只认带引号的值，裸格式被跳过 → **console 显示"关闭"而服务在跑**
- 打码：face-analysis 自己的 `/etc/face-analysis.conf` 里有同名 `BLUR_ENABLED`，
  被 init 脚本翻译成 `--no-blur`，在设备级配置读到之前就关死了

两次都是「只用 API 自测发现不了」—— API 自己写的文件格式当然自洽。
**改配置必须手写一份裸格式测一遍。**

**其他**：
- `/` 是只读 ext4，不 remount 的 `cp` 会失败但脚本继续跑 → 报告成功实际没换
- 重启后 init 自动拉起 face-analysis 占住摄像头和 ION，手动跑别的应用会在 TPU 分配处断言失败
- 手动跑 app 必须 `sudo`，普通用户开不了 `/dev/cvi-tpu0`
- `ps | grep xxx` 会匹配到自己的命令行，造成假阳性（我中过两次）
- 上游 bug：`alpha_factor = 256` 溢出成 0（寄存器 8 位），`force_alpha` 打开时遮罩隐形

---

## 六、下一步

待办列表在会话的 task list 里，摘要：

1. 真机验证 privacy_blur 剩余项（需人出镜）
2. 验证 alpha 可调与黑带修复
3. 验证驱动部署/还原（被 7 阻塞）
4. **按功能拆开提交**（被 1/2/3 阻塞）：内核 / 组件 / 应用接线 / supervisor / console
5. 内核补丁提 PR 给上游（改动只碰 4 文件、无新依赖、不调新 ioctl 时行为不变，形态适合上游）
6. 应用页快捷开关 + manifest 声明检测主体（不涉及人的应用不显示打码入口）
7. 把补丁 `.ko` 打进 supervisor deb —— **现在 console 会在每台设备上报 `not_packaged`，按钮永远是灰的**

另外还有 9 个已构建待发的 deb（版本已 bump，未上 CDN）。

### 两个已知但没处理的问题

- 预测线程每 33ms 移动区域但不更新 LUT，格子数一变驱动就拒绝旧表、退回纯色 → 间歇闪烁
- `mosaic_lut_query()` 是破坏性的，每帧先清表再恢复 → 同上

---

## 七、设备状态

`192.168.42.1`，当前跑 face-analysis，补丁内核已装（`cv181x_rgn.ko` md5 `2d6c6fe3`）。

原厂驱动备份两份：设备 `/userdata/ko-backup/*.orig`、本机 scratchpad `ko_backup/`。
md5 `3934b79e`(rgn) / `5570faa9`(vpss)。**换驱动出问题就从这里还原 + 重启。**
