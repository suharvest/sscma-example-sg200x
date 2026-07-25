# reCamera ONVIF 支持实施说明

> 面向执行者（开发 Agent / 工程师）。目标是让 reCamera 能被主流 VMS 发现、拉流，并以 ONVIF 标准形式输出 AI 分析结果。
>
> **本文的调研结论均已核实到官方规范原文或本地源码，关键处标注了 `file:line` 与规范章节号。执行时如与现场不符，以现场为准并回报。**

---

## 0. 一句话目标

**Profile T 打底（必须满足），Profile M 尽量靠拢（schema 层面对齐，认证暂不做）。**

三个必须先记住的判断：

1. **schema 与认证解耦。** Milestone 的 ONVIF driver 只检查设备上报了 Metadata 能力 + 实现了 Media/Media2，**不检查 Profile 认证标记**。所以"按标准 schema 输出"能拿到全部实际收益，"买认证"只买徽标。
2. **T 的强制项比 M 多。** T 要求 Event/PullPoint（M 里只是 Conditional）、WS-Discovery（M 里只是 Optional）、multicast、Imaging、OSD 等。做 T 的过程中 M 的传输层是白送的。
3. **T 给管子，M 给语义。** T 只要求 metadata 管道里跑 PTZ Status + Property Events，**T 不要求 Analytics**。是 M 定义了 `tt:Object` / `BoundingBox` / `Class` 那套场景描述。

---

## 0.5 🚨 两个前置问题（与 ONVIF 独立，但会阻塞商业发布）

调研过程中发现的两个既有问题，**都不是 ONVIF 引入的，但都会在上 ONVIF 后被放大**。建议在阶段 1 之前决策。

### A. 现网 live555 存在网络可达的高危漏洞

设备上跑的 RTSP 服务，其 live555 是 **2020.07.21**（已实测：`strings libcvi_rtsp.so | grep 20xx.xx.xx` 只有这一个版本，且是**静态**编进 `.so` 的，ELF `DT_NEEDED` 里没有 `libliveMedia.so`）。落后当前版本（2026.07.23）**6 年、约 100 个 release**。

受影响且**全部为网络可达路径**的漏洞：

| 编号 | 说明 | 严重度 |
|---|---|---|
| **CVE-2026-41470** | 拿到有效 Session token 后，可从第二条**未认证** TCP 连接重放 `PLAY`/`TEARDOWN`，导致虚函数调用崩溃或中断他人码流 | **CVSS 4.0 = 8.2 HIGH** |
| CVE-2021-41396 | 短时间大量 socket 连接触发 heap overflow | — |
| CVE-2021-39283 | 多次 SETUP+PLAY 触发 assertion failure 进程退出 | — |
| CVE-2023-37117 | 处理 SETUP 时 heap-use-after-free | — |
| （2026.03.23 修复，无编号） | 已 SETUP 过 RTP-over-TCP 的 session 再收到 non-interleaved SETUP → **use-after-free** | — |

> **一旦上 ONVIF，设备会被 VMS 和资产扫描器纳入摄像头库并主动探测，这些会被直接扫出来。**

**处置**：升级到 live555 2026.07.23（见 §14.2，已实测可干净交叉编译，**体积零增长**）。这件事本身独立于 ONVIF，建议尽快单独排期。

### B. mongoose 是 GPL-2.0-only，与本仓库的 Apache-2.0 不兼容

```
// components/mongoose/mongoose.h 头部（MG_VERSION "7.17"）
SPDX-License-Identifier: GPL-2.0-only or commercial
```

而仓库根 `LICENSE` 是 **Apache-2.0**。

**Apache-2.0 与 GPL-2.0-only 互不兼容**（FSF 明确表态：Apache 2.0 的专利条款被 GPLv2 视为"附加限制"；GPLv3 修好了这点，但 mongoose 标的是 `GPL-2.0-only`）。当前发布的二进制里 Apache-2.0 代码链着 GPL-2.0-only 代码，**两边条款都满足不了**。

**这不是 ONVIF 引入的，是既有状态。** 但它决定了一件对商业模式至关重要的事：

| 许可证 | 你能开源 | **客户能拿去做闭源产品** |
|---|---|---|
| **Apache-2.0**（当前声明） | ✅ | ✅ **可以** |
| GPL-2.0 / GPL-3.0 | ✅ | ❌ **不行** |
| LGPL（live555） | ✅ | ✅ 可以（动态链接 + 不改库源码） |
| **EPL-2.0 OR BSD-3-Clause**（libmosquitto） | ✅ | ✅ 可以（选 BSD-3） |

> **「我们开源」和「客户能闭源」是两回事。** GPL 也是开源，但强制下游同样开源。reCamera 选 Apache-2.0 等于向客户承诺"拿去改、做闭源产品也行"——这是商业模式的基础，不能被 GPL 依赖破坏。

#### 影响面（已实测）

`components/debug_stream` 依赖 mongoose，而它被 **7 个 gallery app** 使用（部分显式在 `PRIVATE_REQUIREDS` 里写了 `mongoose`，部分靠传递依赖，声明本身不一致）。加上 supervisor，**受影响的是所有带 Live 预览的 app**，不是"改一下 supervisor"。

**但好消息**：supervisor 的约 **7700 行业务代码（`api_device.cpp` 1362 行、`api_app.cpp` 1757 行、`api_wifi`、`api_halow` 等）零个 `mg_` 引用**，全部通过 `http_interface.h:21` 的 `typedef const struct mg_http_message* request_t` + 8 个访问器隔离。

真正耦合 mongoose 的只有 **4 个文件 / 约 55 个调用点 / 约 900 行**：

| 文件 | 耦合度 |
|---|---|
| `components/debug_stream/src/debug_stream.cpp` | 重（WS 广播 + 背压 + wakeup） |
| `solutions/supervisor/main/include/http_server.h` | 重（事件循环 + TLS + 静态文件） |
| `solutions/supervisor/main/include/async_exec.h` | 重（wakeup + 连接查找 + 手写 reply） |
| `solutions/supervisor/main/include/http_interface.h` | 中（`request_t` typedef + 3 个访问器） |

#### 选型结论：**libwebsockets（MIT）首选，civetweb 次选**

**civetweb 出局的三个理由（全部已实测/一手核实）：**

1. **master 分支当前编译不过。** 用本项目工具链实测复现：`src/civetweb.c:19211` 括号错位 + 未声明变量 `cl`，引入自 2026-04-19 commit，**已坏 2 个月无人发现**。配合 **3 年 3 个月无 release**（最新 v1.16 / 2023-04）、227 open issue、近一年 55 commit（1/3 是 dependabot）、单人维护——必须 pin 在 2023 年版本，未来安全修复靠自己 backport。
2. **符号碰撞 → 无法渐进迁移。** civetweb 也用 `mg_` 前缀且签名不同（如 `mg_printf`），**同一二进制不可能同时链两个库** → supervisor + debug_stream + 4 个依赖它的 solution 必须一次性大爆炸切换。而 **libwebsockets 用 `lws_` 前缀，可与 mongoose 共存 → 可分两步、各自可回滚**。
3. **架构上正好缺你们最依赖的两样东西**（通读 `civetweb.h` 全部 1834 行 API 确认）：
   - **无 `mg_wakeup()` 等价物** → `async_exec.h` 的 328 行异步/取消/gate 三态语义要整套拆掉重设计
   - **无发送缓冲水位 API**，`mg_websocket_write` 是**阻塞写** → 而 `debug_stream.cpp:133` 正是靠读 `c->send.len` 做慢客户端背压/跳帧/`DS_NEEDS_IDR` 重同步
   
   ⚠️ **第二条要命：迁到 civetweb 等于在 RT VENC 回调线程边界重新引入未验证的阻塞写路径——正是 commit `8b5b65c` 刚修完的那个坑（RT 回调饿死 RTSP）。**

**libwebsockets 的架构是 1:1 映射：**

| 现用 | lws 对应 |
|---|---|
| `mg_mgr_poll` | `lws_service()` |
| **`mg_wakeup()`** | **`lws_cancel_service()`**（头文件明确 "may be called from another thread while the context exists"） |
| `MG_EV_WAKEUP` | `LWS_CALLBACK_EVENT_WAIT_CANCELLED` |
| `c->send.len` 水位 | `LWS_CALLBACK_SERVER_WRITEABLE` + `lws_get_peer_write_allowance()` / `lws_partial_buffered()` — **比现方案更正确** |
| 手写遍历 `mgr->conns` 广播 | `lws_callback_on_writable_all_protocol()` — 内建 |
| `c->data[0]/data[1]` | `per_session_data` |

维护状态：MIT、5.3k star、**每日提交**、多人维护、v4.5.x 持续迭代。

> **核心判断**：civetweb 的优势（官方 `CivetServer` C++ 封装、`document_root` 一行搞定静态文件、`mg_handle_form_request` multipart、体积小 15%、单文件集成、实测 riscv64-musl 零 warning）**都对应迁移里最简单的那 20%**；lws 的优势**对应风险最高的那 80%**。

**工作量估算**：civetweb 12~18 人日（其中 8~10 天集中在高风险重设计）；lws 10~15 人日，风险分布均匀。

#### ✅ 体积实测结果（已完成，用本项目工具链在 `ubuntu_dev_x86` 容器内实测）

libwebsockets **v4.5.8** 交叉编译到 riscv64-linux-musl：**零 warning、零 patch、零 musl 兼容问题**。

| 方案 | text (.a/.o) | 倍数 | 归档/目标文件 |
|---|---|---|---|
| mongoose 7.17（当前配置） | **96,371** | 1.00x | 426,192 B (.o) |
| civetweb v1.16（no TLS + WS） | **110,355** | 1.15x | 512,968 B (.o) |
| lws 配置 A（常规裁剪） | 226,733 | 2.35x | 1,229,854 B (.a) |
| **lws 配置 B（激进裁剪，✅ 采用）** | **190,065** | 1.97x | 1,026,192 B (.a) |
| lws 配置 C（B + `NO_LOGS`） | 180,048 | 1.87x | 966,328 B (.a) |

**链接实测**（100 行 HTTP+WS demo，stripped，动态链 musl）——比看 `.a` 更能反映真实占用：

| 二进制 | text | stripped 文件大小 |
|---|---|---|
| demoB（配置 B） | **145,509** | 150,448 B |
| demoC（配置 C） | 137,387 | 142,248 B |

链接器确实剔除了死代码（`.a` 190KB → 链入 145KB；`nm demoB | grep -icE " lejp_| upng|_inflate"` → `0`）。

> **净代价：比 mongoose 多约 50~80 KB text。**
>
> **判据：原定"超过 500KB text 或 musl 有硬伤则回退 civetweb"——lws 以 145KB 链接体积远低于该线，✅ 通过。**
>
> ⚠️ 不要把这 50~80KB 当作否决理由。设备是 **256MB RAM 的 Linux**，跑 OpenCV + TPU 推理，face-analysis 单是 FairFace 模型就 21MB。真正紧张的是 **ION（~60MB）**，而 ION 服务于 VPSS/VENC/模型，与代码段无关。**更重要的是：换库的驱动因素是许可证不兼容，没有"不换"这个选项，只有"换库"或"买许可"。**

#### ✅ 决策：迁移到 libwebsockets v4.5.8，配置 B

**采用配置 B 而非 C** —— C 的 `LWS_WITH_NO_LOGS=ON` 把 notice/warn 全掐了，设备现场无法排障，只省 8KB，不值。

**构建配置要点**（完整命令行见 §14.8）：

- 必须 `-DLWS_WITH_STATIC=ON -DLWS_WITH_SHARED=OFF`、`-DLWS_WITH_SSL=OFF`、`-DLWS_WITHOUT_CLIENT=ON`
- 关掉全部 event-lib 适配（`LIBUV/LIBEVENT/LIBEV/GLIB/SDEVENT/ULOOP=OFF`），用内建 poll
- 关掉 `HTTP2 / ROLE_MQTT / ROLE_DBUS / ZLIB / SECURE_STREAMS / CONMON / SYS_STATE / SMD / METRICS / NETLINK / LEJP / CBOR / COSE / JOSE / GENCRYPTO / CGI / SPAWN / THREADPOOL` 等 40+ 项

**🚨 两个必踩的构建坑：**

1. **`LWS_WITH_JPEG` 默认 ON，GCC 10.2 下直接编译失败**：
   ```
   lib/misc/jpeg.c:1293:9: error: 'c' may be used uninitialized [-Werror=maybe-uninitialized]
   ```
   GCC 10.2 误报 + lws 自带 `-Werror`。**必须** `-DLWS_WITH_JPEG=OFF -DLWS_WITH_UPNG=OFF -DLWS_WITH_DLO=OFF -DLWS_WITH_LHP=OFF`（lws 4.5 新增的显示/图形栈，本项目无用）。

2. **`LWS_LOG_TAG_LIFECYCLE=OFF` 触发上游 bug**（`lib/core/logs.c:171` unused variable）。而且 **`-DCMAKE_C_FLAGS=-Wno-error` 无效**——lws 在 CMakeLists 里把 `-Werror` 追加在用户 flags **之后**。**只能保持 `LWS_LOG_TAG_LIFECYCLE=ON`。**

**musl 兼容性（实测）**：零问题。`LWS_HAVE_EVENTFD` / `PIPE2` / `TCP_USER_TIMEOUT` / `PTHREAD_SETNAME_NP` 全部探测通过；`LWS_HAVE_NET_IF_ETHER_H` 缺失但 lws 自动回落到 `net/ethernet.h`。GCC 10.2 的 C99/C11 支持完全够用。

#### 🚨 线程模型硬约束（写代码前必读）

已核实 `lws_cancel_service()` 在内建 poll 下**完全可用，且这是它的原生实现路径**（libuv/libevent 反而是后加的适配层）。实现链：

```
lws_cancel_service(ctx)                        [core-net/vhost.c:1109]
  → lws_plat_pipe_signal()                     [plat/unix/unix-pipe.c:69]
  → eventfd_write(pt->dummy_pipe_fds[0], 1)    [已确认走 eventfd 路径]
```
该 fd 被包成一个不绑 vhost/protocol 的特殊 wsi 注册进 pt 的 pollfd 集合，所以 `poll()` 立即唤醒。

> ### ⚠️ **`lws_cancel_service()` 是 lws 唯一保证线程安全、可从工作线程调用的 API。**
> ### **`lws_callback_on_writable()` 不是线程安全的，禁止跨线程调用。**

因此 `debug_stream` 的迁移写法是**固定的**，不允许自由发挥：

```
[推理线程 / VENC 回调线程]
    数据 → 自己的环形缓冲（加锁）
    → lws_cancel_service(ctx)          ← 唯一允许的跨线程调用

[lws 事件循环线程]
    LWS_CALLBACK_EVENT_WAIT_CANCELLED
        → lws_callback_on_writable_all_protocol(ctx, protocol)
    LWS_CALLBACK_SERVER_WRITEABLE
        → 取缓冲；按 lws_get_peer_write_allowance() / lws_partial_buffered()
          决定丢帧 / 置 DS_NEEDS_IDR / 正常发送
```

这套背压模型**比现在读 `c->send.len` 的做法更正确**，但必须严格照此实现——跨线程直接调 `lws_callback_on_writable()` 是最容易踩的坑，且症状是偶发而非必现。

#### 处置路径

| 步骤 | 内容 | 代价 | 状态 |
|---|---|---|---|
| **0** | **与选型无关，建议先做**：把 `debug_stream.cpp:133` 的背压逻辑和 `async_exec` 的 wakeup 语义，在**当前 mongoose 上**先抽象成内部接口（`ws_transport` / `event_poker`）。把后续迁移拆成两个独立可回滚阶段 | 0.5 人日 | 待做 |
| **1** | ~~实测 lws 体积~~ | 1 人日 | ✅ **已完成，通过** |
| **2** | 固化 lws v4.5.8 配置 B 的交叉编译到 `components/libwebsockets/`（脚本化，禁止手工） | 1 人日 | 待做 |
| **3** | 迁 `debug_stream`（WS-only，验证面窄；lws 用 `lws_` 前缀可与 mongoose 共存，**本步可独立上线、独立回滚**） | 3~5 人日 | 待做 |
| **4** | 迁 `supervisor`（`http_interface.h` 适配层 + `http_server.h` 重写 + `async_exec.h` 移植） | 4~6 人日 | 待做 |
| **5** | 移除 `components/mongoose/`，全量回归 8 个 solution 的 deb 打包 + 设备部署 + VPSS 不回归 | 2 人日 | 待做 |
| **并行** | **向 Cesanta 询 mongoose 商业许可报价**。10~15 人日 + 触碰刚修好的并发代码，这个风险是有价格的；若为一次性买断且数字合理，值得比较。（注意：不公开定价、按量报价、需评估供应商锁定） | — | 商务 |

#### 其它候选（已核实，均出局）

| 库 | 出局原因 |
|---|---|
| **lwan** | **GPL-2.0**——同样的许可证问题 |
| **libmicrohttpd** | LGPL-2.1+，嵌入式静态链接麻烦，WS 支持弱 |
| **h2o** | MIT 但依赖多（picotls/quicly/libyaml/OpenSSL），嵌入式过重 |

> ⚠️ **ONVIF 的 HTTP 端点不要继续加深对 mongoose 的依赖**，等本节决策落定再定。这与 §14.4 不推荐 gSOAP 是同一类问题。
>
> 补充：supervisor 的 TLS 当前是**死代码**——`solutions/supervisor/main/CMakeLists.txt:10` 是 `MG_TLS_NONE`，`mg_tls_init()` 在实际固件里是空实现。迁移时不必移植 HTTPS 路径。

---

## 1. 关键前提：先确认目标客户

**这一条会决定阶段 2 做不做，请在开工前向产品确认。**

实测生态支持度（能否解析 ONVIF metadata 并叠加 bbox）：

| VMS | 支持情况 |
|---|---|
| Milestone XProtect | ✅ 原生支持（官方文档确认） |
| Genetec | ✅ 支持 |
| Nx Witness | ⚠️ 6.1 起部分支持，优先做 events，object tracking 排后面 |
| Frigate | ❌ 自己跑 AI，不消费相机 metadata |
| ZoneMinder / Blue Iris / Shinobi | ❌ 只用 RTSP + motion 事件 |

**若目标客户是开源 NVR 阵营，阶段 2（metadata track）投入回报为零，只做阶段 0 + 1 即可。**

---

## 2. 阶段划分与优先级

| 阶段 | 内容 | 依赖 | 何时做 |
|---|---|---|---|
| **0** | ONVIF 数据模型层 + JSON over MQTT | 无 | **立即，零风险** |
| **1** | `components/rtsp_server/` 收敛 + `components/onvif/` 最小服务面 + 集成页 | 阶段 0 的数据模型 | 紧接阶段 0 |
| **2** | metadata RTSP track（`m=application`） | 需换 RTSP 底座 | 目标客户为 Milestone/Genetec 时 |
| **3** | Analytics service 全套 + ONVIF 会员 + DTT 认证 | 阶段 2 | **由订单驱动，不由技术热情驱动** |

---

## 3. 阶段 0：ONVIF 数据模型层 + JSON over MQTT

### 3.1 为什么先做这个

ONVIF 22.12 起**标准化了 metadata 的 JSON 表示和 MQTT topic 结构**（Analytics Service Spec §5.5）。reCamera 现在已经在发 MQTT JSON，把 payload 换个形状就能对上国际标准，**不碰 RTSP、不碰 SOAP、不碰认证**。

更重要的是：这一步产出的「推理结果 → ONVIF 数据模型」序列化层，是**后面所有阶段共用的资产**。XML（阶段 2）和 JSON（阶段 0）只是同一个数据模型的两种输出格式。

### 3.2 架构要求（重要）

```
模型推理结果 (各 app 的私有结构)
      ↓
[ONVIF 数据模型层]  ← 新增，components/onvif_meta/
   ObjectDescriptor / Frame / Appearance ...
      ↓                    ↓
[JSON 序列化]        [XML 序列化]
      ↓                    ↓
   MQTT              RTSP metadata track（阶段 2）
```

**数据模型层必须与传输层解耦。** 不要在 MQTT 发布代码里直接拼 JSON 字符串——阶段 2 要复用同一个模型出 XML。

### 3.3 MQTT Topic 结构（官方 ABNF，Analytics Spec §5.5.2）

```
Topic = TopicPrefix "/" PayloadPrefix "/" MetadataType "/" MetadataProducer

PayloadPrefix    = "onvif-mj"
MetadataType     = "VideoAnalytics" | "AudioAnalytics" | "PTZ" | "SensorData"
MetadataProducer = ProfileToken "/" AnalyticsModuleName
```

官方示例：`MyDevice/onvif-mj/VideoAnalytics/1/MyClassifier`

reCamera 建议取值：`recamera-<sn>/onvif-mj/VideoAnalytics/live0/<app-id>`

> ⚠️ **这是新增 topic，不是替换。** 现有 `recamera/<app>/results` 契约必须保留——`recamera/weather/results` 被 SenseCraft 的 `draw_weather.js` 依赖，改动会破坏线上功能。

### 3.4 XML → JSON 映射规则（官方，规则极简）

- 属性加 `@` 前缀：`UtcTime` → `"@UtcTime"`
- 元素同时有属性和文本时，文本用 `"#text"`
- 可重复元素 → JSON 数组
- 私有命名空间在 `"@context"` 里声明前缀

官方 JSON 样例（Analytics Spec §5.5.3，可直接对照）：

```json
{
  "Frame": [{
    "@UtcTime": "2021-10-05T15:13:27.321",
    "@Source": "MyClassifier",
    "@context": { "acme": "http://www.acme.com/schema" },
    "Transformation": {
      "Translate": { "@x": -1.0, "@y": -1.0 },
      "Scale":     { "@x": 0.003125, "@y": 0.00416667 }
    },
    "Object": [{
      "@ObjectId": 15,
      "Appearance": {
        "Shape": {
          "BoundingBox": { "@left": 20.0, "@top": 80.0, "@right": 100.0, "@bottom": 30.0 },
          "CenterOfGravity": { "@x": 60.0, "@y": 50.0 }
        },
        "Class": { "Type": [ { "@Likelihood": 0.8, "#text": "Human" } ] }
      }
    }]
  }]
}
```

---

## 4. 数据模型规范（阶段 0 和 2 共用）

依据：ONVIF Analytics Service Spec Ver. 26.06 + 官方 XSD `https://www.onvif.org/ver10/schema/metadatastream.xsd`

### 4.1 ⚠️ 坐标系（最容易搞错，必读）

规范：Analytics Spec §5.2.2, Figure 2

- **归一化范围是 `-1..+1`，不是 `0..1`**
- **原点在画面正中心 (0,0)**
- 左 x=-1，右 x=+1；**下 y=-1，上 y=+1 —— y 轴朝上，与图像像素坐标相反**
- 因为 y 朝上，`tt:Rectangle` 必然满足 **`top > bottom`**
- `tt:Rectangle` 的四个属性 `left/top/right/bottom` 全部 required

**模型输出的是 y 向下的像素坐标，直接填会上下颠倒。** 标准做法是在 `tt:Frame` 下加 Transformation 后继续用像素坐标：

```xml
<tt:Transformation>
  <tt:Translate x="-1.0" y="-1.0"/>
  <tt:Scale     x="2/W"  y="2/H"/>
</tt:Transformation>
```

变换公式 `q = p * s + t`。官方例子印证：640×480 → `Scale x="0.003125" y="0.00416667"`（= 2/640, 2/480）。

**注意**：加了这个 Transformation 后，像素坐标仍需 y 翻转（`y' = H - y`），或在 Scale 的 y 用负值并相应调整 Translate。**实现后必须用真实画面验证框的上下位置。**

### 4.2 元素顺序（sequence，顺序错了不合 schema）

`tt:Appearance` 子元素严格顺序：

```
Transformation? → Shape? → Color? → Class? → Extension? → GeoLocation?
→ VehicleInfo* → LicensePlateInfo? → HumanFace? → HumanBody?
→ ImageRef? → Image? → BarcodeInfo? → SphericalCoordinate? → Label*
```

`tt:ShapeDescriptor`：`BoundingBox`(**必填**) → `CenterOfGravity`(**必填**) → `Polygon`* → `Extension`?

> **`CenterOfGravity` 是必填的，不能只给 BoundingBox。**

### 4.3 标准对象类别

`tt:ObjectType` 枚举（10 个）：

```
Animal, HumanFace, Human, Bicycle, Vehicle, LicensePlate, Bike, Barcode, Fire, Smoke
```

`tt:VehicleType`：`Bus, Car, Truck, Bicycle, Motorcycle`

**自定义类别是官方允许的**——`tt:ClassDescriptor/Type` 的类型是 `tt:StringLikelihood`（自由字符串 + Likelihood 属性），不是枚举。XSD 注释原文：*"free type definitions can be added"*。

> ❌ **禁止使用 `tt:ClassCandidate` / `tt:ClassType`** —— XSD 里已标 Deprecated，且该枚举把 Vehicle 拼成了 `Vehical`（ONVIF 历史 typo）。一律用 `tt:ClassDescriptor/Type`。

### 4.4 各 app 的映射方案

| app | 方案 | 状态 |
|---|---|---|
| **yolo-detector / retail-vision** | `tt:Object` + `Shape/BoundingBox` + `Class/Type=Human\|Vehicle` | ✅ 原生场景 |
| **face-analysis** | `Class/Type=Human` + `tt:HumanFace`（`fc:` 命名空间）。FairFace 的性别→`fc:Gender`，年龄→`fc:Age`(Min/Max 区间) | ✅ 好；**emotion 无标准元素**，走私有 ns |
| **qrcode-reader** | `Class/Type=Barcode` + `tt:BarcodeInfo`（`Data`=解码内容，`Type=QRCode`，`PPM`） | ✅ **有官方专用元素，白送** |
| **weather-classifier** | ⚠️ **无场景级分类元素**。变通：造一个覆盖整帧的 Object，`BoundingBox` = (-1, 1, 1, -1)，`Class/Type` 用自由字符串如 `Weather.Rainy` | ⚠️ 变通 |
| **ppocr-reader** | ⚠️ **无通用文本元素**（XSD 里只有 `LicensePlateInfo/PlateNumber` 和 `BarcodeInfo/Data` 能装文本）。走私有命名空间扩展 | ⚠️ 变通 |
| **facemesh-reader** | landmark 无标准表达，走私有 ns；疲劳状态走 Event | ⚠️ 变通 |

> ❌ **分类结果不要用 `tt:Label`。** 那是给 ISO 7010 安全标识牌设计的（Authority 限定 `ISO_3864/ISO_7010/UNECE_ADR/UNECE_GHS`），语义不匹配会误导 VMS。
>
> ❌ **也不要用 `tt:SensorData`**（26.06 新增）——它只支持 `Type` + `float Value`，离散标签塞不进去。

私有扩展的合法位置：每个层级都有 `<xs:any namespace="##any">` 和 `<xs:anyAttribute>`，用自有命名空间即可，官方样例里就是这么干的（`acme:ColorName`）。

建议命名空间：`xmlns:recam="http://www.seeedstudio.com/recamera/schema"`

### 4.5 完整 XML 样例（严格遵守元素顺序，可直接抄）

假定源分辨率 1920×1080。

```xml
<?xml version="1.0" encoding="UTF-8"?>
<tt:MetadataStream
    xmlns:tt="http://www.onvif.org/ver10/schema"
    xmlns:fc="http://www.onvif.org/ver20/analytics/humanface"
    xmlns:bd="http://www.onvif.org/ver20/analytics/humanbody"
    xmlns:wsnt="http://docs.oasis-open.org/wsn/b-2"
    xmlns:tns1="http://www.onvif.org/ver10/topics"
    xmlns:recam="http://www.seeedstudio.com/recamera/schema">

  <tt:VideoAnalytics>

    <!-- 检测 + 人脸属性 -->
    <tt:Frame UtcTime="2026-07-25T03:14:57.321Z" Source="YoloDetector">
      <tt:Transformation>
        <tt:Translate x="-1.0" y="-1.0"/>
        <tt:Scale     x="0.00104166" y="0.00185185"/>
      </tt:Transformation>

      <tt:Object ObjectId="12">
        <tt:Appearance>
          <tt:Shape>
            <tt:BoundingBox left="620.0" top="880.0" right="780.0" bottom="360.0"/>
            <tt:CenterOfGravity x="700.0" y="620.0"/>
          </tt:Shape>
          <tt:Class>
            <tt:Type Likelihood="0.93">Human</tt:Type>
          </tt:Class>
          <tt:HumanFace>
            <fc:Gender>Female</fc:Gender>
            <fc:Age><tt:Min>25</tt:Min><tt:Max>35</tt:Max></fc:Age>
          </tt:HumanFace>
          <!-- 情绪: ONVIF 无对应元素, 私有扩展 -->
          <recam:Emotion Likelihood="0.71">Neutral</recam:Emotion>
        </tt:Appearance>
        <tt:Behaviour>
          <tt:Speed>1.35</tt:Speed>
        </tt:Behaviour>
      </tt:Object>
    </tt:Frame>

    <!-- QR 码: 官方标准表达 -->
    <tt:Frame UtcTime="2026-07-25T03:14:57.621Z" Source="QRCodeReader">
      <tt:Transformation>
        <tt:Translate x="-1.0" y="-1.0"/>
        <tt:Scale     x="0.00104166" y="0.00185185"/>
      </tt:Transformation>
      <tt:Object ObjectId="31">
        <tt:Appearance>
          <tt:Shape>
            <tt:BoundingBox left="820.0" top="700.0" right="1060.0" bottom="460.0"/>
            <tt:CenterOfGravity x="940.0" y="580.0"/>
          </tt:Shape>
          <tt:Class><tt:Type Likelihood="0.99">Barcode</tt:Type></tt:Class>
          <tt:BarcodeInfo>
            <tt:Data Likelihood="0.99">https://wiki.seeedstudio.com/recamera/</tt:Data>
            <tt:Type Likelihood="0.99">QRCode</tt:Type>
            <tt:PPM>4.8</tt:PPM>
          </tt:BarcodeInfo>
        </tt:Appearance>
      </tt:Object>
    </tt:Frame>

    <!-- 全局分类模型兜底: 覆盖整帧的 Object, 归一化坐标不加 Transformation -->
    <tt:Frame UtcTime="2026-07-25T03:14:58.000Z" Source="WeatherClassifier">
      <tt:Object ObjectId="90">
        <tt:Appearance>
          <tt:Shape>
            <tt:BoundingBox left="-1.0" top="1.0" right="1.0" bottom="-1.0"/>
            <tt:CenterOfGravity x="0.0" y="0.0"/>
          </tt:Shape>
          <tt:Class>
            <tt:Type Likelihood="0.82">Weather.Rainy</tt:Type>
            <tt:Type Likelihood="0.13">Weather.Cloudy</tt:Type>
          </tt:Class>
        </tt:Appearance>
      </tt:Object>
    </tt:Frame>

  </tt:VideoAnalytics>
</tt:MetadataStream>
```

### 4.6 空帧与心跳

Analytics Spec §5.4：**即使没检测到任何对象，也应规律发送空的 scene description**，用于告知客户端分析引擎存活。收到 `SetSynchronizationPoint` 请求时**必须**发。

---

## 5. 阶段 1：组件化与集成页

### 5.1 先做 `components/rtsp_server/` 收敛

**现状（已核实）**：8 个 solution 各揣一份 `rtsp_demo.c`，MD5 全部为 `3603ba3c…`，**字节级相同**；8 份 `.h` 也完全相同。detection-blur 那份（`abb2c9e9…`）唯一差异是删了装饰性注释块和两行死代码，**零功能差异**。

抽取时注意：

- `app_ipcam_Rtsp_Server_Create` / `app_ipcam_rtsp_Server_Destroy`（`rtsp_demo.c:83`、`:113`）是非 static 但未在头文件声明的符号
- 全局变量 `stRtspCtx` / `pstRtspCtx`（`rtsp_demo.c:21-22`）
- 以上均需改 `static` 防符号冲突
- `PARAM_CFG_INI` 宏（`rtsp_demo.c:8`）定义了从未使用，删除

**目标 API 以 `ma::TransportRTSP` 的能力集为准，不是以 rtsp_demo 为准。** retail-vision 走的是这条路（`solutions/retail-vision/main/main.cpp:341-352`），它具备 ONVIF 需要而 rtsp_demo 没有的能力：

| 能力 | rtsp_demo（8份） | TransportRTSP |
|---|---|---|
| 端口 | 硬编码 8554（`rtsp_demo.c:36`） | 可配 |
| session 名 | 硬编码 `live%d`（`rtsp_demo.c:142`） | 可配 |
| **鉴权** | **无** | user/pass（live555 UserAuthenticationDatabase） |
| 可关闭 | 无 | `--no-rtsp` |
| 码率 | 硬编码 30720（`rtsp_demo.c:39`） | 可配 |

ONVIF 的 `GetProfiles` / `GetStreamUri` 要回答的正是端口、session、鉴权这些，**新组件必须能自我描述**。

> ⚠️ 组件设计时把「流媒体服务」抽象出来，不要把 cvi_rtsp 的 API 泄漏到组件接口上。阶段 2 要换底座，届时 8 个 app 不应受影响。

### 5.2 组件模板（照抄 `components/ha_mqtt/CMakeLists.txt`，全文 8 行）

```cmake
file(GLOB ONVIF_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

component_register(
    COMPONENT_NAME onvif
    SRCS ${ONVIF_SRCS}
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE_REQUIREDS mongoose pthread
)
```

目录布局（与现有组件一致）：

```
components/<name>/
  CMakeLists.txt
  include/<name>.h     <- 唯一对外头文件，扁平
  src/<name>.cpp
```

solution 侧只需在 `main/CMakeLists.txt` 的 `PRIVATE_REQUIREDS` 加组件名。组件自动被发现，顶层无需注册（驱动逻辑在 `cmake/project.cmake`）。

### 5.3 配置链路：走 `ha.conf` 模式，不走 `/etc/<app>.conf`

**已核实的关键事实**：supervisor 的 HA 配置**完全绕开了** `/etc/<app>.conf` + init 脚本那条链。它写 `/userdata/local/ha.conf`，app 侧 `components/ha_mqtt` 在**进程内直接读**（`components/ha_mqtt/src/ha_mqtt.cpp:62`）。

**照此实现 ONVIF**：

- supervisor 写 `/userdata/local/onvif.conf`
- `components/onvif` 自己 `loadOnvifConfig()`
- **不用改任何 app 的 init 脚本，不用改任何 app 的 `main.cpp` 的 `long_options`/switch/config 结构体**

对比走 `/etc/<app>.conf`：每个 app 改三处 × 8 个 app，且 supervisor 没有写 `/etc/*.conf` 的现成代码。**明确选前者。**

配置文件格式（照抄 `ha_config.h:13-14` 的约定）：`KEY='value'` 逐行，shell-sourceable，单引号包裹，内嵌单引号写成 `'\''`。

Keys：`ONVIF_ENABLED`(0/1) / `ONVIF_PORT` / `ONVIF_USERNAME` / `ONVIF_PASSWORD` / `ONVIF_DEVICE_NAME`

**原子写流程必须照抄 `ha_config.cpp:129-166`**：

```
create_directories → unlink(tmp) → open(tmp, O_WRONLY|O_CREAT|O_TRUNC, 0600)
→ fchmod(fd, 0600) → write → fsync → close → 短写检测 → rename(tmp, CONF_FILE)
```

`load()` 的容错约定（`ha_config.cpp:80-82`, `:97-100`）：**文件缺失即返回默认值，永不算错**；单行解析失败只 LOGW 跳过。

### 5.4 后端 API

照抄 `solutions/supervisor/main/src/api_app.cpp:1428-1534` 的 `setHaConfig`。**以下几处不能漏**：

| 行 | 内容 | 为什么不能漏 |
|---|---|---|
| `:1437-1438` | load 现值 → copy 成 next | 未提供的字段（如 password）自动保留 |
| `:1496-1499` | `op_guard g; acquire_op_or_busy(res, g)` | 与 switchApp 同级互斥，防止写配置与切换 app 竞态 |
| `:1512-1517` | Node-RED 模式短路 `in_nodered_mode()` | 配置照存但不重启 app，回 `restarted:false, note:"nodered_mode"` |
| `:1520-1534` | 读 state → `active_app` → `load_manifests()` → `restart_after_change()` | 使配置生效 |
| `:1408` | 序列化只回 `password_set` | **永不回明文密码** |

路由注册在 `api_app.cpp:121-123` 旁加：

```cpp
REG_API(getOnvifConfig);
REG_API(setOnvifConfig);
```

`REG_API` 宏定义在 `api_base.h:28-33`，URI 自动为 `api/appMgr/<handler名>`（prefix 来自 `api_app::api_app() : api_base("appMgr")`，`api_app.cpp:101-102`），**默认需要 token**。

无需改 `http_server.h`——`_apis` 列表已含 `api_app`。

### 5.5 前端集成页

**扩展点已经留好了**，`solutions/supervisor/www/src/views/integrations/index.tsx:375-381`：

```tsx
/**
 * Integration cards shown on the page, in order.
 * Add a new integration by appending a { key, Card } entry here.
 */
const integrationCards: { key: string; Card: () => JSX.Element }[] = [
  { key: "home-assistant", Card: HomeAssistantCard },
];
```

append 一项即可。**路由和侧边菜单都不用改**（`router/index.tsx:68-71`、`layout/menu.ts:85-87` 保持原样）。

可复用的 UI 零件（同文件内）：`CopyButton`(`:29-60`)、`CodeBlock`(`:62-69`)、RTSP URL 展示块(`:351-359`)、表单+Test/Save 双按钮(`:325-338`)、`Collapse` 折叠高级选项(`:306-323`)。

改动清单：

| # | 文件 | 动作 |
|---|---|---|
| 1 | `www/src/api/onvif/onvif.d.ts` | 新建，照 `api/ha/ha.d.ts:7-51` |
| 2 | `www/src/api/onvif/index.ts` | 新建，照 `api/ha/index.ts:12-56`；set 的 timeout 给 45000（会重启 app） |
| 3 | `www/src/views/integrations/index.tsx` | 加 `OnvifCard` 组件（放 `:373` 附近） |
| 4 | 同上 `:379-381` | 数组 append `{ key: "onvif", Card: OnvifCard }` |
| 5 | `www/src/locales/en-US.json` | 在 `ha` 块（`:564`）后追加 `onvif` 块 |
| 6 | `www/src/locales/zh-CN.json` | 同上（`:559`），**key 必须与 en-US 完全对齐** |

集成页文案要点（给用户看的）：ONVIF service URL、支持的 profile、用户名密码设置、以及**明确写"ONVIF-compatible"而非"ONVIF conformant"**（见 §8 合规）。

### 5.6 最小 ONVIF 服务面（阶段 1 范围）

目标：**被主流 VMS 发现并拉到流**。实现以下即可：

- **WS-Discovery**（UDP 组播 `239.255.255.250:3702`）——Probe / ProbeMatches / Hello
- **Device Management**：`GetServices` / `GetServiceCapabilities` / `GetDeviceInformation` / **`GetSystemDateAndTime`（必须允许匿名访问，见 §5.9）** / `GetScopes` / `GetNetworkInterfaces`
- **`SetSystemDateAndTime`**（转调 `settimeofday()` + `hwclock -w`，见 §5.9）
- **Media2 (ver20)**：`GetProfiles` / `GetStreamUri` / `GetSnapshotUri` / `GetVideoSourceConfigurations` / `GetVideoEncoderConfigurations`
- **Events**：`CreatePullPointSubscription` / `PullMessages` / `SetSynchronizationPoint` / `GetEventProperties` / `Unsubscribe`
- **Digest 认证**（HTTP + RTSP）
- `/snapshot.jpg` HTTP 端点

> ⚠️ **即使阶段 1 也要照 Profile T 的形状做，不要照 Profile S。**
>
> **不做**：MJPEG、WS-UsernameToken、WS-Base-notification（这三个是 S 的强制项而 T 不要求）。
> **要做**：Digest 认证、Media2（不是 Media1）。
>
> 原因：ONVIF 官方已宣布 **Profile S 于 2027-03-31 停止新产品认证**，为 S 写的代码是纯浪费。

**实现方式见 §14.4：手写 XML 模板，禁止用 gSOAP（许可证否决项）。** 动手前先完成 §0.5-B 的 mongoose 决策。

#### 组件拆分：发现是共用的，分析结果不是

这两半的耦合度完全不同，**不要合成一个组件**：

| | `components/onvif_service`（待做） | `components/onvif_meta`（已做） |
|---|---|---|
| 内容 | WS-Discovery / Device / Media2 / Events | 分析结果载荷 |
| app 需要写代码吗 | **完全不需要，链上即可用** | 每个 app 约 5 行 |
| 为什么 | 它回答的全是设备级问题：SN、RTSP 端口、快照地址、网卡、时钟。**跟当前跑的是 yolo 还是 weather 无关** | 「我这个框在 ONVIF 里叫什么类」没有通用答案：face 是 `HumanFace`，yolo 要把 COCO 名映射到 `Human`/`Vehicle`/`Animal`，weather 压根没有框 |

`onvif_service` 要问的每个问题，本轮已经把 API 备齐了：

| ONVIF 操作 | 数据来源 |
|---|---|
| `GetProfiles` / `GetStreamUri` | `rtsp_server_port()` / `_session_name()` / `_auth_enabled()` / `_url()` |
| `GetSnapshotUri` | `debug_stream` 的 `/snapshot.jpg` |
| `GetDeviceInformation` | `ha_mqtt::readDeviceIdentifier()` |
| `GetSystemDateAndTime` | 系统时钟，见 §5.9 |

#### 🚨 默认开启发现 ⇒ 必须同时决定 RTSP 鉴权

**决定：自动发现默认开启**（ONVIF 相机的市场预期，默认关等于没人找得到）。保留 `ONVIF_DISCOVERY_ENABLED` 开关但**缺省为开**（配置文件不存在即视为开，与 `ONVIF_META_ENABLED` 相反），用于两种真实场景：企业网络禁止未批准的服务广播；现场已有别的 ONVIF 网关，重复广播会让 VMS 看到重复设备。

**已决定：选 B —— 发现默认开，RTSP 维持无鉴权。**

设备当前 RTSP 8554 完全无鉴权，WS 8001 亦然。发现服务不增加暴露面——端口本来就开着——但它把"局域网里有台相机、流在这个地址"主动广播出去（UDP 组播 239.255.255.250:3702），从"扫端口才能发现"变成"打开 VMS 自动跳出来"。产品判断是接受这个状态。

**这个选择的最大好处是零破坏性变更**：那 30+ 处硬编码 `rtsp://{host}:8554/live0` 的地方——5 个 app manifest 的 `rtsp_url`、十几份 console 里展示的 app 说明文档（中英各一份）、各 app 的启动日志、README——**一处都不用改**。

**技术上完全可行**：VMS 调 `GetStreamUri` 拿到 URL 直接连，服务端不发起挑战就直接播。VMS 通常会把为 SOAP 配置的凭据一并带上，无鉴权的 RTSP 服务器照收不误。ODM / Milestone / Genetec 均正常。

> ⚠️ **唯一的硬边界**：Profile T §7.1 把 Digest 鉴权列为 Mandatory（HTTP + RTSP + RTSP-over-HTTP 三处都要）。所以在这个配置下**不得对外宣称 Profile T 合规**。这与 §8.1 已定的"只用 ONVIF-compatible 措辞、不碰 logo"一致，不构成新增约束。

**能力已就位，随时可开**：`rtsp_server_config_t` 已有独立的 `username` / `password` 字段，`rtsp_server_auth_enabled()` 可查询，`GetStreamUri` 的应答会自动带上凭据（见 `rtsp_server_url()`）。将来要开只需填配置，不用改接口。

**将来真要默认开启鉴权时，需要连带解决的**（记录在此免得重新推导）：

| 问题 | 说明 |
|---|---|
| 密码从哪来 | 固定出厂默认 = 等于没有鉴权（必然进文档和扫描器字典）；要求用户自设 = 出厂状态拉不了流 |
| 推荐路径 | 复用 console 登录凭据——用户首次登录本就被要求改密码，改完自动具备 RTSP 凭据，既无出厂默认也不用记新密码 |
| 但要注意 | 那等于每接一个 VMS 就把管理员密码交出去。行业惯例是单独的 onvif 账号，权限仅拉流。`rtsp_server` 的接口已经支持拆分，不用改结构 |
| 连带工作量 | 上述 30+ 处 URL 引用要一起更新，并写进 release note |

### 5.7 `/snapshot.jpg` 实现要点

**现状**：目前没有任何 app 维护"标注后的 cv::Mat"。视频流是**未标注的原始 H.264**，检测框由 debug_stream 的 JSON 通道发给前端、浏览器 overlay 绘制。帧生命周期极短——`retrieveFrame` 拿到后用完立刻 `returnFrame`，之后内存失效。

唯一现成的 JPEG 编码路径在 facemesh-reader：

- 抓帧拷贝（**必须在 `returnFrame` 之前**）：`solutions/facemesh-reader/main/main.cpp:445-457`，关键是 `wrap.clone()`（`:454`）
- 编码段（**完全通用，建议抽公用**）：`main.cpp:399-405`
  ```cpp
  cvtColor(annotated, bgr, COLOR_RGB2BGR);
  resize(bgr, resized, cv::Size(640, 360));
  cv::imencode(".jpg", resized, jpeg, {cv::IMWRITE_JPEG_QUALITY, 80});
  ```
- 节流参考：`main.cpp:363-373` `snapshot_cooldown_ok()`

**改造要求**：facemesh 现在只在 HA 模式 + 报警边沿触发时抓（`main.cpp:449-450`）。ONVIF 需要"始终保留最近一帧"。**按 debug_stream 的惰性策略做**（参考 `debug_stream.cpp:308` 的 `video_clients > 0` 判断），只在有订阅时才 clone + 编码。

### 5.8 🚨 线程纪律（这是踩过坑的地方）

`components/debug_stream` 已有一个 mongoose `mg_mgr` 在 8001 端口，poll 循环是：

```cpp
// components/debug_stream/src/debug_stream.cpp:198-201
static void ds_poll_loop(void) {
    while (g_ds.running.load(std::memory_order_acquire)) {
        mg_mgr_poll(&g_ds.mgr, 100);
    }
}
```

**技术上可以在同一个 mgr 上挂 ONVIF 路由**（在 `ds_ev_handler` 的 `:184` 404 兜底之前插分支），但必须遵守：

1. **绝对禁止在 poll 线程里做 `cv::imencode`**。生产者线程预编码好 JPEG 存进 mutex 保护的 latest-snapshot buffer，poll 线程只做 `mg_http_reply` 吐 buffer。
2. **绝对禁止在 poll 线程里做重量级 XML 解析**。SOAP 处理要么足够轻，要么另起自己的 mgr + 线程。
3. 跨线程唤醒**只能**用 `mg_wakeup(&mgr, listener_id, ...)`，生产者线程**从不直接碰 mongoose 对象**（这条纪律在 `components/debug_stream/include/debug_stream.h:15-17` 已明确写了）。
4. 若在同 mgr 上再 `mg_http_listen` 第二个端口，注意 `g_ds.listener_id` 是单值假设（`debug_stream.cpp:267`），需要区分。

**推荐做法**：`components/onvif` 起自己的 mgr + 线程。本仓库既定风格就是"每模块一个 mgr"——supervisor 自己也是独立的（`http_server.h:94`，250ms 周期）。复制 debug_stream 的模式成本很低。

> **历史教训**：本项目发生过「CH 回调在实时优先级上跑重活，饿死 RTSP 服务线程，端口绑了但不 accept」的故障。ONVIF 挂到共享 poll 线程会继承同类风险，表现是"VMS 能发现设备但取不到流"。

### 5.9 时钟与 RTC（实测结论，修正了早期的错误判断）

> **早期版本的本文写过"设备无 RTC"。这是错的。** 实测结论如下。

#### 现状：三层都有问题，但最容易修的是软件那层

| 层面 | 状态 |
|---|---|
| SoC 片内 RTC 模块 | ✅ 有（RTCSYS 子系统，`RTCSYS_CORE @ 0x05026000`） |
| 驱动 + 编译产物 | ✅ 有，`/mnt/system/ko/cv181x_rtc.ko` 就在设备上 |
| 设备树节点 | ✅ 有，`/proc/device-tree/rtc`，platform device `5026000.rtc` |
| `CONFIG_RTC_CLASS` | ✅ `=y` |
| **`insmod`** | ❌ **`loadsystemko.sh:26` 被注释掉了** ← **根因** |
| 备份电池 / 超级电容 | ❌ 无，也无预留焊盘 |
| 32.768kHz 晶振给 RTC | ❌ 无（板上那颗是给 WiFi 的）→ 走 **32Kless** 模式，片内 RC + 软件校准 |

**实测加载即可用**（约 5 分钟，无需重编内核/改设备树）：

```sh
insmod /mnt/system/ko/cv181x_rtc.ko
# dmesg:
#   cvi_rtc 5026000.rtc: registered as rtc0
#   cvi_rtc 5026000.rtc: rtc 32k calibration has been completed
#   cvi_rtc 5026000.rtc: CVITEK real time clock
hwclock -w -f /dev/rtc0   # 写入（注意 busybox hwclock 默认找 /dev/misc/rtc，必须给 -f）
hwclock -r -f /dev/rtc0   # 读回
```

写入/读回/计数已实测全部正确。

#### 掉电行为

`+VDD_RTC` 的唯一来源是核心板 **U9（XC6206P182MR 固定 1.8V LDO）**，输入是 VIN，**无二极管 OR、无电池切换电路**（原理图确认）。

| 情况 | 结果 |
|---|---|
| `reboot` / 软复位 / 内核重启 | ✅ **时间保持**（RTC 域只要 VIN 在就持续供电） |
| 拔电 / 断 VIN | ❌ 秒计数器归零 → 回到 1970 |

> 顺带澄清一个误导性网络名：`PWR_VBAT_DET`(SG2002 pin 38) **不是 RTC 电池**，原理图注释写明 "When VBAT_DET < (1.0V) shutdown"，是系统输入欠压检测（R8/R9 从 VIN 分压）。

#### 🚨 启用 RTC 的两条风险

**1. 加载顺序——这条要认真对待。**

注册 RTC 设备时内核会做 hctosys（RTC → 系统时钟）：

```
cvi_rtc 5026000.rtc: setting system clock to 1970-01-01T00:00:00 UTC (0)
```

冷启动 RTC 为 0，系统时钟被设成 1970。**当前不构成回归**（设备本来就停在 1970），**但如果 NTP 先同步成功、RTC 模块后加载，系统时钟会被倒拨 56 年**——时间倒流会连锁破坏文件 mtime、日志顺序、TLS 证书校验和任何依赖单调时间的逻辑。

> ✅ **正确做法：恢复 `loadsystemko.sh` 那行注释的原位**（启动早期，远早于网络就绪）。
> ❌ **不要在 supervisor 的 S93 init 里补 `insmod`**——那个位置太靠后，有倒拨风险。

**2. 精度——RC 振荡器不是晶振。**

`rtc 32k calibration has been completed` 说明走的是 32Kless 模式。系统运行时的时钟走 25MHz 主晶振（准），RTC 计数器走片内 RC（不准），**两者会渐行渐远**。重启后 `hwclock -s` 拿到的是 RC 那份漂过的时间。

**实测漂移**（25 分钟基线）：

```
T0:  sys=1784958851   RTC=13:54:11     ← hwclock -w 对齐
T1:  sys=1784960363   RTC=14:19:22     ← 1512 秒后
     设备系统时钟 == Mac 参考（完全一致，走 25MHz 主晶振）
     RTC 应读 14:19:23，实读 14:19:22 → 慢 1 秒
```

**≈ 660 ppm ≈ 每天 57 秒。**

> ⚠️ 精度限制：`hwclock` 是 1 秒分辨率，25 分钟基线的量化误差本身就是 ±660ppm 量级。**这个数只能说明"数百 ppm 量级"**，点估计 660ppm；要更准需要数小时基线。但对决策已经足够。

**含义**：设备若在上次 NTP 之后运行一天再重启，RTC 给出的时间可能差约一分钟。**ONVIF digest 的 5 秒窗口，RTC 单独扛不住。**

#### 不构成风险的（已逐条核实）

- "RTC invalid time" 日志噪音 —— cosmetic，每次冷启动一条
- 校准耗时 —— 实测 **71ms**，可忽略
- 掉电唤醒被误开 —— 驱动只在设置闹钟时才置 `EN_PWR_WAKEUP |= 0x30`，不设闹钟不碰
- 与其他模块冲突 —— 实测加载后 qrcode-reader / supervisor / VPSS 全部正常，dmesg 无异常
- 掉电后回 1970 —— 与现状完全一致，不是回归

#### 对 ONVIF 的结论

**RTC 解决的是"重启后不至于回到 1970"，不解决"时间足够准"。** 真正的解法是规范本身给的：

1. **`GetSystemDateAndTime` 必须实现且允许匿名访问**（§5.6 已列为强制项）。ONVIF 规定客户端应先调它拿设备时间，**用设备时钟而非本地时钟计算 digest nonce** —— 规范兼容的客户端（ODM、主流 VMS）会自动补偿偏差，**即便设备停在 1970 也能通过认证**。这是成本最低、最鲁棒的解。
2. **实现 `SetSystemDateAndTime`**，内部转调 `settimeofday()` + `hwclock -w`，让客户端首次连接时把时间灌进来。supervisor 已有现成的 `settimeofday` 接口（`api_device.cpp:560-593`，吃 `timestamp` 参数）。
3. RTC 是锦上添花：加载驱动后，supervisor 现有的 NTP 链路**立刻自动生效**——`main.sh:280` 那行 `hwclock -w >/dev/null 2>&1 || true` 写了很久但一直静默失败（没有 `/dev/rtc0`，错误被 `|| true` 吞掉）。

---

## 6. 阶段 2：metadata RTSP track

### 6.1 为什么需要换 RTSP 底座

**已核实**：`cvi_rtsp` 的结构体硬编码死了两条 track。

```c
// sg2002_recamera_emmc/cvi_rtsp/install/include/cvi_rtsp/defs.h
typedef struct {
    CVI_RTSP_TRACK video;
    CVI_RTSP_TRACK audio;
    char name[128];
} CVI_RTSP_SESSION;
```

codec 枚举也只有 `RTSP_VIDEO_{H264,H265,JPEG}` 和 `RTSP_AUDIO_{PCM_L16,PCM_L24,AAC}`。**从 C 接口这一层伸不出第三条腿。**

同时 Profile T 的其它强制项 cvi_rtsp 也不暴露：**RTP/UDP Multicast**（T §7.9 Mandatory）、RTSP-over-HTTP 隧道、Digest 认证配置。

### 6.2 两条路

| | 方案 A：fork cvi_rtsp | 方案 B：自建 live555 server |
|---|---|---|
| 改动量 | 小（几十行 diff） | **比预想小得多——见下** |
| 上层影响 | 几乎不用动 | 可通过兼容层做到零改动 |
| ABI 风险 | **高**——改结构体即改内存布局，设备 rootfs 有原版 `libcvi_rtsp.so`，`ma_transport_rtsp` 也链它 | 无 |
| 维护 | 要长期维护 fork | 自有代码 |
| multicast / HTTP 隧道 / Digest | 仍需自己往 fork 里加洞 | live555 内建 |
| 线程模型 | 仍藏在库里 | **可控**（可根治 §5.8 的历史问题） |
| live555 版本 | 绑死 2020.07.21（含 §0.5-A 的 CVE） | 可用 2026.07.23 |

### **结论：方案 B。** 上游调研推翻了"fork 更省事"的假设

1. **`sophgo/cvi_rtsp` 不是 CV181X 的代码。** 它的首个 commit 是 `cvi_rtsp for bm1688 v1.5`，`cvi_models/` 里放的是 `*_cv186x.bmodel`——这是 **BM1688 / CV186X（SOPHON 线）**。CV181X 线的源码只存在于 `scpcom/cvi_rtsp` 的 `cv18xx-v4.2.x` 镜像分支，最后更新 **2025-03-14**。
2. **仓库没有 LICENSE 文件**，13 个 commit 全是 "upload 1.7/1.8/2.0" 式整包投递，无真实开发历史。fork 它 = 背上法律不明 + 维护双重债务。
3. **`CVI_RTSP_SESSION` 两年半没动过。** 已 diff 验证：上游 master、`scpcom` cv18xx-v4.2.x、`DangNgoHai04/sg2002-cvi_rtsp` 三个分支的 `defs.h` 与本地**字节级相同**。没有任何 fork 加过 metadata track。
4. **要改的正是它最核心的数据结构**，fork 后与上游 diff 立刻不可 rebase，"跟上游"这个唯一好处当场消失。
5. **🔑 核心代码只有约 28KB。** `src/api.cpp` 6.9KB + 7 个 `.hpp`（`cvi_rtsp.hpp` 5.1KB / `cvi_smss.hpp` 7.7KB / `cvi_video_smss.hpp` 2.9KB / `cvi_audio_smss.hpp` 3.1KB / `cvi_jpeg_source.hpp` 4.0KB / `cvi_source.hpp` 3.0KB / `ring_buffer.hpp` 2.4KB），本质就是 live555 `OnDemandServerMediaSubsession` + `FramedSource` 的一层薄封装。**自己重写约 1-2 天**，比 fork 划算。

**兼容层要求**：新组件对外导出同名 `CVI_RTSP_*` API，`CVI_RTSP_SESSION` 增加 `CVI_RTSP_TRACK metadata` 字段。现有 solutions 零改动迁移。

> ⚠️ **参考实现**：`scpcom/cvi_rtsp` 的 `cv18xx-v4.2.x` 分支（`src/cvi_smss.hpp` / `cvi_video_smss.hpp` / `ring_buffer.hpp`）。**读它、理解思路、自己重写——该仓库无 LICENSE 文件，禁止直接复制粘贴代码。**

### 6.3 live555 实现要点

Ross Finlayson（live555 作者）在邮件列表给出的官方做法：

1. 写一个 `FramedSource` 子类，重写 `doGetNextFrame()`，吐出 `tt:MetadataStream` XML 片段
2. 写一个 `OnDemandServerMediaSubsession` 子类，重写 `createNewStreamSource()` 和 `createNewRTPSink()`
3. `createNewRTPSink()` 里直接用现成的 `SimpleRTPSink`：

```cpp
SimpleRTPSink::createNew(envir(), rtpGroupsock, rtpPayloadTypeIfDynamic,
                         90000, "application", "vnd.onvif.metadata");
```

`SimpleRTPSink` 会自动生成正确的 SDP 行。

**唯一有技术风险的点**：ONVIF 要求 **RTP marker bit = 1 表示 XML 文档结束**，而 `SimpleRTPSink` 默认在最后一个 fragment 置 marker。若一个 XML document 跨多个 RTP 包，可能要重写 frame 边界逻辑。**实现时需实测验证。**

### 6.4 承载规范（Streaming Spec Ver. 26.06 §5.1.2.1.1 / §5.2.2.4）

SDP：

```
m=application 0 RTP/AVP 107
a=control:rtsp://<host>/onvif_camera/metadata
a=recvonly
a=rtpmap:107 vnd.onvif.metadata/90000
```

- encoding name 三选一：`vnd.onvif.metadata`（未压缩）/ `vnd.onvif.metadata+gzip`（payload 以 RFC1952 头开始）/ `vnd.onvif.metadata.exi.ext`
- 时钟率固定 **90000**
- Payload type 用动态范围 **96–127**
- **marker bit = 1 表示 XML 文档结束**
- XML 文档大小无限制，**建议最长每 1 秒开一个新文档**
- 收到 `SetSynchronizationPoint` 时**必须**关闭当前文档、开新文档

---

## 7. 阶段 3：认证（由订单驱动）

Profile T 剩余强制项差距：

| 项 | 难度 |
|---|---|
| **Media2 全套（~30 个 operation）** | 🔴 最大工程量，数周 |
| **Event service 完整**（PullPoint + ItemFilter/TopicFilter + ≥2 并发订阅 + ProfileChanged/ConfigurationChanged 事件） | 🔴 高 |
| Imaging service（Get/Set/Options） | 🟡 可复用现有 camera.conf / ISP 基础 |
| OSD（CreateOSD/DeleteOSD/GetOSDOptions/SetOSD） | 🟡 CVI RGN 已有能力，接 API |
| RTP/UDP Multicast | 🟡 live555 支持 |
| Tampering + MotionAlarm 事件 | 🟢 现有 CV 能力足够 |
| WS-Discovery + Scope 增删改查 | 🟢 |
| Digest auth | 🟢 live555 内建 |
| HTTPS/TLS | ⚪ Conditional，可免 |

### 🚨 Conditional 陷阱（必读）

Profile T 规范 §5 原文：

> Conditional = Feature that shall be implemented … **if it supports that functionality in any way, including any proprietary way.**

**不是"想做才做"，而是"只要你用任何私有方式实现了，就必须用 ONVIF 方式暴露"。** 对照 reCamera：

| Conditional 项 | reCamera 现状 | 结果 |
|---|---|---|
| §8.9 音频输入 | 板载 mic，`arecord -D hw:0,0` | **变强制** |
| §8.12 音频输出/双向音频 | 外接喇叭，`aplay -D hw:1,0` | **变强制** |
| §8.14 Focus Control | 已有 FV 对焦辅助 | **变强制** |
| §8.6 Motion Region Detector | retail-vision `config_schema` 有 `zone`/`line` 类型 | **大概率变强制** |
| PTZ | B401 云台版 | **该 SKU 变强制** |

**"没这个硬件所以免掉"的空间比预期小得多。** 唯一确定能免的是 HTTPS/TLS。

### 认证成本

- **必须是 ONVIF 会员才能声明 conformance**
- 会员年费：Full $20,000 / Contributing $10,000 / Registered Affiliate $5,000 / **User $4,000** / Observer $500
- **最低门槛 = User 会员 $4,000/年**
- Observer（$500）**能拿 DTT 自测但明确禁止对外宣称任何合规性**
- DTT（Device Test Tool）只在 Member Portal 提供，每年发布两次（6月/12月）
- 流程是自声明：跑 DTT → 工具自动生成 DoC + Feature List + Test Report → 手写 ONVIF Interface Guide → 提交。**无第三方实验室环节**

---

## 8. 合规红线

### 8.1 ONVIF 商标

- **未取得会员资格前，禁止使用 "ONVIF conformant" / "ONVIF certified" 措辞，禁止使用 ONVIF/Profile 徽标图形**
- 安全措辞：`ONVIF-compatible`、`interoperable with ONVIF clients`、`implements a subset of ONVIF Profile T`
- 官方原文："Products claiming improper conformance are asked to cease claims until conformance can be met"
- **产品文案定稿前请法务过一遍**

### 8.2 live555 许可（LGPLv3）

live555 **仍是 LGPL v3 or later**（已逐字核对 2026.07.23 tarball：每个 `.cpp/.hh` 头部写明 "either version 3 of the License, or (at your option) any later version"；`COPYING` = GPLv3 全文，`COPYING.LESSER` = LGPLv3）。

live555 官方 FAQ 明确的三条义务：

1. **改了 `.cpp/.hh` 就必须公开你的修改**——发到 live-devel 邮件列表**不算**履行义务，必须在产品官网上放出来
2. **只做 C++ 子类化、不改原始文件 → 你的子类和应用代码可以闭源**。官方原话：*"If, instead, you subclass the supplied code (without modifying it), you are not required to release your subclass code (nor the rest of your application code). Your application can be 'closed source'."*
3. 必须给用户"替换库代码"的能力 → 实践上意味着**必须动态链接 `.so`**，且**固件必须可升级**。FAQ 明确写：没有固件升级机制的硬件产品**不能使用本软件**

| 链接方式 | 义务 |
|---|---|
| **动态链接 .so** | LGPLv3 §4(d)(1)"suitable shared library mechanism"——只需附许可证副本 + 显著声明。**必须走这条** |
| 静态链接 .a | §4(d)(0)——必须提供 **Minimal Corresponding Source（你自己的 .o 目标文件）**。商业产品通常不可接受 |

> ⚠️ **当前实现已存在合规缺口**：`libcvi_rtsp.so` 把 live555 **静态**编了进去（实测 ELF `DT_NEEDED` 里没有 `libliveMedia.so`），用户无法替换 live555 —— 不满足 LGPL relinking 要求。
>
> **改成"外部 `libliveMedia.so` + 只做子类化"可同时解决合规、安全（§0.5-A 的 CVE）和升级三个问题。**

必做合规动作：① 动态链接 ② **一行不改 live555 源码，只做子类化** ③ 随产品附 LGPL/GPL 许可证副本 ④ 在文档或"关于"页声明使用了 live555 ⑤ 保证固件可升级。

> Live Networks 提供商业许可，无公开价目表，需联系 `support@live555.com` 面谈。

### 8.3 GenICam / GenTL

（备查）GenICam（EMVA）授权**免费**，无 royalty——若将来评估机器视觉方向可参考。

---

## 9. 代码位置速查

### 现有可复用资产

| 用途 | 位置 |
|---|---|
| 组件 CMake 模板 | `components/ha_mqtt/CMakeLists.txt`（8 行） |
| 配置文件原子写 | `solutions/supervisor/main/src/ha_config.cpp:129-166` |
| 配置读取容错 | `ha_config.cpp:76-125` |
| app 侧进程内读配置 | `components/ha_mqtt/src/ha_mqtt.cpp:62` |
| API handler 模板 | `solutions/supervisor/main/src/api_app.cpp:1428-1534` |
| API 路由注册 | `api_app.cpp:121-123`，宏在 `api_base.h:28-33` |
| 集成页扩展点 | `www/src/views/integrations/index.tsx:375-381` |
| mongoose mgr + poll 线程模式 | `components/debug_stream/src/debug_stream.cpp:252-285` |
| 跨线程唤醒纪律 | `components/debug_stream/include/debug_stream.h:15-17` |
| JPEG 编码段 | `solutions/facemesh-reader/main/main.cpp:399-405` |
| 帧拷贝（returnFrame 前） | `solutions/facemesh-reader/main/main.cpp:445-457` |
| 惰性策略参考 | `components/debug_stream/src/debug_stream.cpp:308` |
| RTSP 高级封装（目标 API） | `components/sscma-micro/porting/sophgo/ma_transport_rtsp.h:20-66` |
| manifest `{host}` 替换 | `api_app.cpp:666-675`（注意注释里的线程约束） |
| manifest 加载 | `api_app.cpp:368-400` |

### manifest 扩展

`load_manifests()` **整体透传 JSON，未知字段自动带过去，不需要改**。加 `onvif` 字段直接写进 `solutions/<app>/rootfs/usr/share/<app>/<app>.json` 即可。

若 URL 用 `{host}` 占位，需在 `api_app.cpp:666-675` 旁加同样的替换逻辑。**注意该处注释的线程约束：`get_host(req)` 必须在 poll 线程上读，`req` 只在当前 event 有效。**

---

## 10. 护栏

### 禁止事项

- ❌ 禁止 `rm -rf` 任何非 build 产物目录
- ❌ 禁止修改 `/etc/init.d/` 下 usb0 / 网络相关脚本（USB 192.168.42.1 是唯一的恢复通道）
- ❌ 禁止改动现有 MQTT topic 契约，尤其 `recamera/weather/results`（SenseCraft `draw_weather.js` 依赖）
- ❌ 禁止在 mongoose poll 线程做 JPEG 编码或重 XML 解析
- ❌ 禁止使用 `tt:ClassCandidate` / `tt:ClassType`（deprecated）
- ❌ 禁止对外使用 "ONVIF conformant" 措辞
- ❌ 禁止静态链接 live555

### 构建与部署

- 构建只能用各 solution 的 `deploy.sh`，或 CLAUDE.md 中的 Docker 一键命令，**禁止裸调 cmake/make**
- `control/{postinst,prerm,...}` 在 git 里必须是 `+x`，否则 opkg 报 126。Docker Desktop 挂载缓存会吃掉 chmod，重启容器再打包
- 部署前确认设备只有一个 app 在用摄像头（VPSS 独占）

### 交付要求

每个阶段的报告必须包含 **EVIDENCE 段**：

- 每个产物的 md5
- 功能验证的**原始工具输出**（不是摘要）
- `dmesg | grep -E 'vpss|fail|error|Oops'` 筛查结果
- before/after 对照

---

## 11. 验收标准

### 阶段 0

- [ ] `mosquitto_sub` 能收到 `recamera-<sn>/onvif-mj/VideoAnalytics/live0/<app>` 的 JSON
- [ ] payload 通过 ONVIF JSON 形状检查（`Frame` 数组、`@` 前缀属性、`#text`）
- [ ] 原有 `recamera/<app>/results` topic 输出**逐字节未变**
- [ ] bbox 坐标验证：用已知位置的目标核对归一化值，**特别确认上下方向没有颠倒**

### 阶段 1

- [ ] ONVIF Device Manager（免费工具）能发现设备并显示设备信息
- [ ] VLC 能通过 ONVIF 返回的 `GetStreamUri` 拉到流
- [ ] `/snapshot.jpg` 返回有效 JPEG，且**连续请求不影响视频 WS 帧率**（对比测试）
- [ ] Digest 认证生效（错误密码被拒）
- [ ] 集成页开关能正确写入 `/userdata/local/onvif.conf` 并重启 app 生效
- [ ] 关闭开关后 ONVIF 端口不再监听
- [ ] **`GetSystemDateAndTime` 匿名可调**（不带凭据也返回 200），且设备时钟停在 1970 时 ONVIF Device Manager 仍能通过 digest 认证（见 §5.9）

### 阶段 2

- [ ] `DESCRIBE` 响应的 SDP 中出现 `m=application` + `a=rtpmap:<pt> vnd.onvif.metadata/90000`
- [ ] Wireshark 抓包确认 marker bit 在 XML 文档结束时置 1
- [ ] XML 通过官方 XSD 校验（`metadatastream.xsd`）
- [ ] `SetSynchronizationPoint` 后立即收到新文档
- [ ] 无检测目标时仍规律收到空帧
- [ ] **在真实 Milestone XProtect 上能看到 bbox 叠加**（这是阶段 2 唯一有意义的验收）

---

## 12. 未决问题

**开工前必须由产品/法务决策的（阻塞项）：**

1. **目标客户是 Milestone/Genetec 还是开源 NVR？** —— 决定阶段 2 是否启动
1b. ~~RTSP 是否默认启用鉴权？~~ —— **已决定：不开**，见 §5.6 末。零破坏性变更；代价是不得宣称 Profile T 合规（§7.1 要求 Digest），这与既定的 "ONVIF-compatible" 措辞一致
2. **mongoose GPL-2.0 如何处置？**（§0.5-B）—— 技术选型已收敛到 **libwebsockets**（civetweb 已因 master 编译不过 + 符号碰撞不可渐进 + 缺 wakeup/背压 而出局），待决的是：**迁移 vs 买 Cesanta 商业许可**。**不决策则 ONVIF 做完也无法商业发布**，且当前 Apache-2.0 与 GPL-2.0-only 的组合本身不合规
3. **live555 升级是否单独排期？**（§0.5-A）—— 现网有 CVSS 8.2 的网络可达漏洞，独立于 ONVIF

**技术层面待实测确认的：**

4. `SimpleRTPSink` 的 marker bit 语义与 ONVIF "XML 文档结束"是否一致（跨包场景）—— 阶段 2 实测
5. Profile T 的 `SetSystemFactoryDefault` / `SystemReboot` 与 supervisor 现有恢复出厂/重启逻辑如何对接
6. Conditional 陷阱中的音频项：是否要为 ONVIF 补齐音频流（现有 RTSP 的 `#ifdef AUDIO_SUPPORT` 分支各 solution 均未启用）
7. ~~设备无 RTC~~ —— **已查清，见 §5.9。结论修正：设备有片内 RTC，只是驱动被注释掉了。** 剩余待定项是 RC 振荡器的实际漂移量（测量中）和是否推动上游恢复 `insmod`

---

## 13. 参考规范

| 文档 | 用途 |
|---|---|
| [ONVIF Profile T Specification v1.0](https://www.onvif.org/wp-content/uploads/2018/09/ONVIF_Profile_T_Specification_v1-0.pdf) | 强制项清单 |
| [ONVIF Profile M Specification v1.1](https://www.onvif.org/wp-content/uploads/2024/04/onvif-profile-m-specification-v1-1.pdf) | metadata 语义 |
| [ONVIF Analytics Service Spec Ver. 26.06](https://www.onvif.org/specs/srv/analytics/ONVIF-Analytics-Service-Spec.pdf) | §5 场景描述、§5.5 JSON/MQTT |
| [ONVIF Streaming Spec Ver. 26.06](https://www.onvif.org/specs/stream/ONVIF-Streaming-Spec.pdf) | §5.2.2.4 metadata RTP 承载 |
| [metadatastream.xsd](https://www.onvif.org/ver10/schema/metadatastream.xsd) | **规范性 schema，元素顺序以此为准** |
| [common.xsd](https://www.onvif.org/ver10/schema/common.xsd) | Rectangle / Vector / Polygon 定义 |
| [Profile Feature Overview v2.6](https://www.onvif.org/wp-content/uploads/2022/04/onvif-profile-feature-overview.pdf) | S/T/M 对照表 |
| [Milestone ONVIF driver — Metadata](https://doc.milestonesys.com/latest/en-US/onvifdriver/metadata.htm) | 消费端要求 |
| [Axis Scene Metadata over RTSP](https://developer.axis.com/analytics/axis-scene-metadata/how-to-guides/scene-metadata-over-rtsp/) | 最贴近标准的公开实现参考（Axis 是 Profile M 主要起草方） |

---

## 14. 依赖与 SDK 选型

### 14.1 SDK 版本现状：升级 SDK 对本需求毫无帮助

本地 SDK **几乎可确定是 reCamera-OS 0.2.1（2025-10-11）**——`cvi_rtsp/install/lib/*` 的 mtime 全是 `Oct 11 2025`，与 0.2.1 发布日吻合；0.2.1 release notes 的 "Increase ION size to 60M" 也与已知的 ION ~60MB 上限一致。

最新是 **0.2.4（2026-03-23）**。但 0.2.2~0.2.4 的 release notes 全文只有：文件浏览器、SSH 开关、Node-RED 升级、CDC、wifi halow、电池电压检测、ISP CSIBDG fifo overflow 修复——**零 RTSP 相关条目**，`cvi_rtsp` header 也没变。

> **结论：不需要为 ONVIF 升级 SDK。**（ISP fifo overflow 修复可能对 VPSS 稳定性有价值，但那是独立议题。）

上游状态速查：

| repo | 分支 | 最近 push |
|---|---|---|
| `sophgo/sophpi` | `sg200x-evb` | 2026-07-02 |
| `sophgo/cvi_mpi` | `sg200x-dev` | 2026-06-30 |
| `sophgo/host-tools` | `master` | 2025-09-26（工具链仍是 **GCC 10.2.0**） |

### 14.2 live555：自编 2026.07.23，出动态 `.so`

**决定**：不用 SDK 里那份 2020.07.21（CVE 见 §0.5-A，静态链接不合规见 §8.2）。

⚠️ **下载地址已迁移**：`http://www.live555.com/liveMedia/public/` 现在返回 **404**。新址是 **https://download.live555.com/**

**已实测**：用现有 reCamera 工具链（`riscv64-unknown-linux-musl-g++`，Xuantie-900 V2.6.1，GCC 10.2.0）**0 warning 编译通过**。

产物体积对比（**升级零体积代价**）：

| 库 | 2026.07.23 | SDK 的 2020.07.21 |
|---|---|---|
| `libliveMedia.a` | 4,585,492 | 4,501,896 |
| `libgroupsock.a` | 193,016 | 181,654 |
| `libBasicUsageEnvironment.a` | 130,294 | 130,222 |
| `libUsageEnvironment.a` | 20,608 | 20,036 |

tarball 自带 30 个 `config.*` 但**没有 riscv**。可用的 `config.riscv-musl`：

```make
CROSS_COMPILE?=		riscv64-unknown-linux-musl-
COMPILE_OPTS =		$(INCLUDES) -I. -O2 -DSOCKLEN_T=socklen_t -D_LARGEFILE_SOURCE=1 \
			-D_FILE_OFFSET_BITS=64 -DNO_OPENSSL=1 -DNO_STD_LIB=1 -fPIC
C =			c
C_COMPILER =		$(CROSS_COMPILE)gcc
C_FLAGS =		$(COMPILE_OPTS) $(CPPFLAGS) $(CFLAGS)
CPP =			cpp
CPLUSPLUS_COMPILER =	$(CROSS_COMPILE)g++
CPLUSPLUS_FLAGS =	$(COMPILE_OPTS) -std=c++17 -Wall -DBSD=1 $(CPPFLAGS) $(CXXFLAGS)
OBJ =			o
LINK =			$(CROSS_COMPILE)g++ -o
LINK_OPTS =		-L. $(LDFLAGS)
CONSOLE_LINK_OPTS =	$(LINK_OPTS)
LIBRARY_LINK =		$(CROSS_COMPILE)ar cr␣          # ← 末尾必须有一个空格
LIBRARY_LINK_OPTS =
LIB_SUFFIX =		a
LIBS_FOR_CONSOLE_APPLICATION =
LIBS_FOR_GUI_APPLICATION =
EXE =
```

然后 `./genMakefiles riscv-musl && make -j8`。

**三个必踩的坑：**

1. **`-DNO_STD_LIB=1` 是必需的。** 官方 `config.linux` 现在用 `-std=c++20`，而 `BasicUsageEnvironment0.hh:114` 用了 `std::atomic_flag::test()`——**libstdc++ 11 才有**，GCC 10.2.0 会报 `'struct std::atomic_flag' has no member named 'test'`。加 `-DNO_STD_LIB` 后重跑 `genMakefiles`。加了之后 `-std=c++17` 完全够用（已 grep 确认全树无 `string_view`/`concept`/`consteval`/`<=>`），与本项目 C++17 规范一致。
2. **`LIBRARY_LINK` 行尾的空格不能少**——Makefile 模板直接拼接库名，少空格会变成 `ar crlibliveMedia.a` 并报一大段 usage。
3. `config.linux` 默认带 `-lssl -lcrypto`。SDK sysroot 有 OpenSSL 1.1（`libssl.so.1.1`）；只做明文 RTSP 就 `-DNO_OPENSSL=1` 省体积，要 RTSPS/SRTP 再去掉。

**交付要求**：把交叉编译脚本（`config.riscv-musl` + genMakefiles 调用）固化到 `components/live555/`，**禁止手工操作**。产出 4 个 `.so` 放进 rootfs，应用只做子类化。

### 14.3 RTSP server：自建（见 §6.2 的完整论证）

### 14.4 ⛔ SOAP 层：手写 XML 模板，**禁止用 gSOAP**

**gSOAP 开源版对闭源商业产品是直接否决项。**

gSOAP 的 `LICENSE.txt` 明确排除：

> "Components NOT covered by the gSOAP Public License are: — wsdl2h tool **AND its source code output**, — soapcpp2 tool **AND its source code output**, ..."

而 ONVIF 开发流程必然是 `wsdl2h onvif.wsdl → soapcpp2` 生成几十万行 stub。**这些 stub 只能按 GPLv2 使用**。官方原话：

> "If you use gSOAP under the GPL v2 to integrate parts of it **or code generated by it** with your own code ... you must make the source code of your programs available to the users of your programs."

**即：用开源版 gSOAP 做 ONVIF，整个 reCamera 应用（含算法、模型加载逻辑）必须以 GPLv2 开源。** Genivia 官网甚至主动提醒 "Black Duck Scans will detect the use of GPL gSOAP software in your project builds"。

商业许可（Standard / Enterprise 两档，royalty-free EULA）无公开价格，需 `contact@genivia.com` 询价。且 ONVIF 全量 stub 通常 5-15MB 源码、编译后 1-3MB，对存储紧张的 CV181X 不友好。

> 顺带排除 `roleoroleo/onvif_srvd`：它自己标 BSD-3-Clause，但**底下的 gSOAP stub 仍是 GPLv2**，等于假 BSD。

**推荐做法**（clean-room 实现）：

- **响应**：为每个 ONVIF operation 准备一份 XML 模板文件，运行时做 `%KEY%` 占位符替换。**ONVIF 响应结构完全固定，模板化后根本不需要 XML 序列化库**
- **请求解析**：只需从 SOAP Body 取出 operation 名 + 少量参数，SAX 回调抓取即可，不建 DOM
- **认证**：Digest / WS-UsernameToken（Digest + Nonce + Created）用 SDK 已有的 OpenSSL 1.1 做 SHA-1/Base64，无需额外引入 mbedtls
- **HTTP 端点**：优先 **civetweb（MIT）** 或 lighttpd + CGI，**避免继续加深对 GPL-2.0 mongoose 的依赖**（见 §0.5-B）
- **WS-Discovery**：独立 UDP 3702 多播守护进程，几百行，与 SOAP 层解耦

**架构参考**：`roleoroleo/onvif_simple_server`（88★，2026-07-16 仍活跃，支持 **Profile S + T**，C 语言 + ezxml + XML 模板，无 gsoap 无 libxml，以 CGI 形式跑在 busybox httpd 后面）。

> ⚠️ **该项目是 GPL-3.0，只能读架构、理解思路，禁止复制代码。**

商业替代（若时间压力大且预算允许）：**Happytime ONVIF Server**——支持 S/T/G/M/C/A 全 Profile，核心二进制 **~300KB**，零第三方依赖（内建 XML/HTTP/SOAP parser），提供 C 源码，支持交叉编译。价格需询价。

### 14.5 XML 库：bundle expat 2.8.2 静态链接

| 库 | 版本 | 许可 | 评价 |
|---|---|---|---|
| **expat** ✅ | 2.8.2（2026-06-25） | **MIT** | SAX 流式，纯 C，musl/RISC-V 零适配；解析峰值内存 O(单元素)，最省 RAM |
| pugixml | 1.16 | MIT | DOM，速度最快但内存 ≈ 文档大小 × 1.5 |
| tinyxml2 | 11.0.0 | Zlib | 单文件 DOM，性能/内存逊于 pugixml |
| ezxml | 停更多年 | MIT | 46KB 单文件，**对不可信输入不建议** |

> ⚠️ **禁止链 sysroot 里那份 expat**：`buildroot .../usr/lib/pkgconfig/expat.pc` 显示 **`Version: 2.4.1`**（`libexpat.so.1.8.1`）。NVD 对 expat 2.4.1 记录了 **46 条 CVE**（整数溢出、UAF、堆溢出、栈耗尽、hash flooding）。ONVIF 端点解析的是**未认证网络攻击者可控的 XML**，用 2.4.1 等于开门。
>
> **必须自己 bundle 2.8.2 并静态链进应用**（MIT 许可无冲突），同时避免污染 rootfs。

若采用"响应纯模板 + 请求只抓 operation 名"的方案，甚至只用 expat 的解析侧，序列化侧一个库都不需要。

### 14.6 ONVIF WSDL / XSD 获取（全部免费、无需注册，已逐个验证 HTTP 200）

| 服务 | URL | 大小 |
|---|---|---|
| Device Management ver10 | `https://www.onvif.org/ver10/device/wsdl/devicemgmt.wsdl` | 195 KB |
| **Media2 ver20** | `https://www.onvif.org/ver20/media/wsdl/media.wsdl` | 149 KB |
| **Analytics ver20** | `https://www.onvif.org/ver20/analytics/wsdl/analytics.wsdl` | 33 KB |
| Events ver10 | `https://www.onvif.org/ver10/events/wsdl/event.wsdl` | 47 KB |
| Imaging ver20 | `https://www.onvif.org/ver20/imaging/wsdl/imaging.wsdl` | 26 KB |
| PTZ ver20 | `https://www.onvif.org/ver20/ptz/wsdl/ptz.wsdl` | 61 KB |
| 公共 schema | `https://www.onvif.org/ver10/schema/onvif.xsd` | 422 KB |

ONVIF 允许自由复制分发（保留 copyright / license / disclaimer）。

### 14.7 建议的实施顺序（依赖层面）

1. **固化 live555 2026.07.23 交叉编译** → `components/live555/`（已验证可行，**1 天**）
2. **自建 `components/rtsp/`**，API 兼容 `CVI_RTSP_*` 并新增 metadata track，同时修掉 §5.8 的 RT 回调饿死 RTSP 老问题（**3-5 天**）
3. **最小 demo 验证三 track**（video H.264 + audio + `application/vnd.onvif.metadata`）能被 ONVIF Device Manager / VLC 正确 SETUP（**1 天**）
4. **再动 SOAP 层**——但在此之前**必须先完成 §0.5-B 的 mongoose 许可证决策**，否则 ONVIF 做完仍然无法商业发布

### 14.8 libwebsockets v4.5.8 交叉编译配方（已实测，可直接复现）

**环境**：容器 `ubuntu_dev_x86`，`/workspace` → `/Users/harvest/project/recamera`
**工具链**：`riscv64-unknown-linux-musl-gcc` (Xuantie-900 V2.6.1 B-20220906) **GCC 10.2.0**
**版本**：`git clone --depth 1 -b v4.5.8`（commit `fbb0baf`）

**CMake toolchain file**（`tc.cmake`）：

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(TC /workspace/host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-)
set(CMAKE_C_COMPILER   ${TC}gcc)
set(CMAKE_CXX_COMPILER ${TC}g++)
set(CMAKE_AR           ${TC}ar     CACHE FILEPATH "")
set(CMAKE_RANLIB       ${TC}ranlib CACHE FILEPATH "")
set(CMAKE_STRIP        ${TC}strip  CACHE FILEPATH "")
set(CMAKE_C_FLAGS_INIT "-march=rv64imafdcv0p7xthead -mcpu=c906fdv -mabi=lp64d -O2")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

**配置 B 完整命令行**（采用的配置）：

```bash
cmake ../libwebsockets \
 -DCMAKE_TOOLCHAIN_FILE=./tc.cmake -DCMAKE_BUILD_TYPE=Release \
 -DLWS_WITH_SSL=OFF -DLWS_WITHOUT_TESTAPPS=ON -DLWS_WITH_MINIMAL_EXAMPLES=OFF \
 -DLWS_WITH_HTTP2=OFF -DLWS_ROLE_MQTT=OFF -DLWS_ROLE_DBUS=OFF \
 -DLWS_ROLE_RAW_PROXY=OFF -DLWS_ROLE_RAW_FILE=OFF \
 -DLWS_WITH_ZLIB=OFF -DLWS_WITH_ZIP_FOPS=OFF -DLWS_WITH_HTTP_STREAM_COMPRESSION=OFF \
 -DLWS_WITHOUT_CLIENT=ON \
 -DLWS_WITH_LIBUV=OFF -DLWS_WITH_LIBEVENT=OFF -DLWS_WITH_LIBEV=OFF \
 -DLWS_WITH_GLIB=OFF -DLWS_WITH_SDEVENT=OFF -DLWS_WITH_ULOOP=OFF \
 -DLWS_WITH_STATIC=ON -DLWS_WITH_SHARED=OFF \
 -DLWS_WITH_JPEG=OFF -DLWS_WITH_UPNG=OFF -DLWS_WITH_DLO=OFF -DLWS_WITH_LHP=OFF \
 -DLWS_IPV6=OFF -DLWS_WITH_PLUGINS=OFF \
 -DLWS_WITH_SYS_ASYNC_DNS=OFF -DLWS_WITH_SYS_NTPCLIENT=OFF -DLWS_WITH_SYS_DHCP_CLIENT=OFF \
 -DLWS_WITH_CONMON=OFF -DLWS_WITH_SYS_STATE=OFF -DLWS_WITH_SYS_SMD=OFF \
 -DLWS_WITH_SYS_METRICS=OFF -DLWS_WITH_SECURE_STREAMS=OFF \
 -DLWS_WITH_NETLINK=OFF -DLWS_WITH_UDP=OFF \
 -DLWS_WITH_LEJP=OFF -DLWS_WITH_LEJP_CONF=OFF -DLWS_WITH_LWSAC=OFF \
 -DLWS_WITH_STRUCT_JSON=OFF -DLWS_WITH_CBOR=OFF -DLWS_WITH_COSE=OFF \
 -DLWS_WITH_JOSE=OFF -DLWS_WITH_GENCRYPTO=OFF \
 -DLWS_WITH_CACHE_NSCOOKIEJAR=OFF -DLWS_WITH_HTTP_UNCOMMON_HEADERS=OFF \
 -DLWS_WITH_CUSTOM_HEADERS=OFF -DLWS_WITH_ACCESS_LOG=OFF -DLWS_WITH_RANGES=OFF \
 -DLWS_WITH_CGI=OFF -DLWS_WITH_SPAWN=OFF -DLWS_WITH_PEER_LIMITS=OFF \
 -DLWS_WITH_SYS_FAULT_INJECTION=OFF -DLWS_WITHOUT_EXTENSIONS=ON \
 -DLWS_WITH_HTTP_BASIC_AUTH=OFF -DLWS_WITH_HTTP_DIGEST_AUTH=OFF -DLWS_WITH_HTTP_PROXY=OFF \
 -DLWS_WITH_THREADPOOL=OFF -DLWS_WITH_DIR=OFF -DLWS_WITH_FTS=OFF \
 -DLWS_WITH_DISKCACHE=OFF -DLWS_WITH_LWS_DSH=OFF -DLWS_WITH_SUL_DEBUGGING=OFF \
 -DLWS_WITH_TLS_SESSIONS=OFF \
 -DLWS_LOG_TAG_LIFECYCLE=ON     # ← 必须 ON，否则 logs.c:171 -Werror 失败
```

> **注意**：`LWS_WITH_LEJP=OFF` / `LWS_WITH_UPNG=OFF` 并不完全生效——`lejp-conf.c.o`(10.7KB)、`upng-gzip.c.o`(8.3KB) 仍被编进 `.a`。但链接器会丢掉（已用 `nm` 验证），只是 `.a` 虚胖，无实际代价。
>
> **裁剪已见底**：从配置 A 到 C 关掉 40+ 选项，text 只从 226KB 降到 180KB。剩下的是 lws 架构成本（vhost/wsi/role/pt 抽象）：`server.c` 17.8KB、`libwebsockets.c` 14.9KB、`parsers.c` 10.2KB、`header.c` 9.0KB、`vhost.c` 8.9KB、`wsi.c` 8.9KB……不可能再压。

**交付要求**：把 `tc.cmake` + 上述命令行固化到 `components/libwebsockets/` 的构建脚本里，**禁止手工操作**。

**未验证项**：容器内无 qemu-riscv64，demo 只做了交叉编译 + 链接验证（ELF EXEC / RISC-V 已确认），**未在真机运行**。阶段 3 迁移 `debug_stream` 时需真机验证。
