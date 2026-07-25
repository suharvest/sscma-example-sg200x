# mongoose → libwebsockets 迁移进度

> 活文档，随迁移推进更新。背景与决策见 `docs/onvif-implementation-spec.md` §0.5-B（为什么必须迁）、§14.8（构建配方）、§14.10（工作量拆解）。

**驱动因素**：mongoose 是 `GPL-2.0-only`，仓库是 Apache-2.0，两者不兼容。这既是当前发布的合规问题，也决定了**客户能不能基于 reCamera 做闭源产品**——GPL 会强制下游开源。

---

## 进度总览

| # | 步骤 | 状态 | 实际人日 | commit |
|---|---|---|---|---|
| 0 | 三层抽象（前置） | ✅ 真机验证 | — | `8c87c40` `cd53813` `4828fc3` |
| 1 | 固化 lws 交叉编译 | ✅ 验证 | 0.5 | `bb1e591` |
| 2 | `ws_transport_lws`（debug_stream 后端） | ✅ **真机 16/16** | 1.5 | 见下 |
| 3 | `http_request_lws`（请求侧） | ⬜ | — | — |
| 4 | `http_dispatch_lws`（回复侧） | ⬜ | — | — |
| 5 | `http_server_lws`（事件循环宿主） | ⬜ | — | — |
| 6 | 移除 mongoose + 全量回归 | ⬜ | — | — |

**可分两批独立上线**（lws 用 `lws_` 前缀，能与 mongoose 共存）：
- 批次 A = 步骤 2 → 只影响 7 个 app 的 Live 预览，可独立验证与回滚
- 批次 B = 步骤 3-5 → supervisor，含 console 静态资源与文件上传

---

## 步骤 0 留下的资产

耦合面已收敛到 **3 个后端适配器 + 1 个事件循环宿主**，业务代码（约 7700 行）零改动：

```
components/debug_stream/src/ws_transport_mongoose.cpp   → 待加 _lws 兄弟文件
solutions/supervisor/main/include/http_dispatch_mongoose.h
solutions/supervisor/main/include/http_request_mongoose.h
solutions/supervisor/main/include/http_server.h          → 需重写
```

**接口当初就是照 lws 的形状设计的**（"lws 是更受限的那个：禁止跨线程碰连接、没有窥探 socket 发送缓冲的原语；mongoose 能满足这些约束，反过来不行"）。所以 `debug_stream.cpp` / `async_exec.h` / 全部业务代码在迁移中**一行不动**。

**验收基线**（步骤 0 真机实测，迁移后需复现）：

| 项 | mongoose 实测值 |
|---|---|
| WS 测试套件 | 16/16 通过 |
| 慢客户端背压：健康客户端吞吐 | 20 秒 / 201 帧 / 185KB |
| 慢客户端背压：最坏帧间隔 | **0.12s**（均值 0.100s） |
| 404 响应体 | `not found\n`（10 字节） |
| 客户端上限 | 第 3 个视频客户端 503 |
| `/snapshot.jpg` 冷启动 | 503 → armed 后 200 |

压测脚本：`ws_verify.py`（16 条断言，见 `8c87c40` 的验证记录）。

---

## 步骤 1：lws 构建（已完成）

- 版本 **v4.5.8**，`components/libwebsockets/fetch_and_build.sh` 可复现
- 产物不入库（68MB 源码树不适合 vendoring，对比 mongoose 是单文件）
- **190,113 B text**，与选型实测 190,065 一致
- 烟测：context 创建 → `lws_cancel_service()` → service → 销毁，链接后 **145,069 B text**

**两个构建坑**（都伪装成项目自己的问题）：
1. `LWS_WITH_JPEG` 默认 ON，GCC 10.2 误报 + lws 自带 `-Werror` → 必须关掉（连带 UPNG/DLO/LHP）
2. `LWS_LOG_TAG_LIFECYCLE=OFF` 触发上游 bug，且 `-Wno-error` 无效（lws 把 `-Werror` 追加在用户 flags 之后）→ 必须保持 ON

---

## 步骤 2：ws_transport_lws（进行中）

### 与 mongoose 的实质差异

| ws_transport 接口 | mongoose | lws |
|---|---|---|
| `ws_conn_send()` | `mg_ws_send()` **同步写进连接缓冲** | **不替应用缓冲**：入队 + `lws_callback_on_writable(wsi)`，真正写发生在 `SERVER_WRITEABLE` 回调 |
| `ws_conn_backlog()` | 读 `c->send.len` | 自维护队列的字节数 |
| `ws_transport_for_each()` | 遍历 `mgr->conns` | lws 无公开的"遍历全部 wsi"，需自维护连接表 |
| tag slots | `c->data[0..1]` | `per_session_data` |
| `ws_transport_wake()` | `mg_wakeup()` | `lws_cancel_service()`（**唯一线程安全的跨线程 API**） |
| `on_drain` 触发 | `MG_EV_WAKEUP` | `LWS_CALLBACK_EVENT_WAIT_CANCELLED` |

### mongoose 没有的额外约束

- **每个待发缓冲前必须预留 `LWS_PRE` 字节**给 lws 写帧头，不能发裸 buffer
- **`lws_callback_on_writable()` 不是线程安全的**——只能在事件线程调用。这条已写进 `ws_transport.h` 的契约

### 实测结果：16/16，与 mongoose 基线数字一致

| 项 | mongoose | lws 4.5.8 |
|---|---|---|
| 慢客户端饿死 20s，健康客户端 | 201 帧 | **201 帧** |
| 最坏帧间隔 | 0.12s | **0.10s** |
| 均值间隔 | 0.100s | **0.100s** |

### 踩到的三个坑（都不是"换个函数名"级别）

**1. `LWS_PROTOCOL_LIST_TERM` 在 C++ 下不可用**（用了 designated initializer）。改用显式 `{ nullptr, ... }` 终止符。C 的烟测编得过，所以这个坑要到编 C++ 才暴露。

**2. `LWSMPRO_CALLBACK` 的 mount 必须同时设 `origin` 和 `protocol`。** 只设 `protocol` 能编译、能跑到 `lws_create_context`，然后**解引用 NULL 崩溃**——现象是内核寄存器 dump（`cause: 0xd` 载入页错误、`badaddr: 0x1e`），不是构建错误也不是断言。lws 自己的示例两个都设。

**3. 🚨 SDK 里有三份 libwebsockets，版本不一致**：

```
sysroot/usr/include                 lws 4.0.22 头   ← project.cmake:16，include 路径第 1 位
cvitek_tpu_sdk/include              lws 4.1.7  头   ← 第 2 位
sysroot/usr/lib/libwebsockets.so    4.0.22
cvitek_tpu_sdk/lib/libwebsockets.a  4.1.7
```

曾一度改用 SDK 那份（理由是"省一次构建、避免双库共存"），结果**编译用 4.0.22 的头、链接 4.1.7 的库**：两者 `LWS_WITH_*` 启用项数不同（21 vs 18），`lws_context_creation_info` 字段偏移不同，赋值落到错误成员上，启动即崩。

**结论反转：私有 pin 版比 SDK 版更安全**，因为头和库严格配套，且 `include_directories(BEFORE)` 能同时压过 SDK 的两份。顺带还小（190KB vs 321KB text，SDK 那份带着用不到的 HTTP/2 + TLS + client）。

### 一个测试环境教训

中途出现 11/17，怀疑是计数逻辑 bug，加了两轮 trace 才发现：**之前 playwright 打开的 Live Debug 页面一直连着**，占了 1 个视频 + 1 个结果名额，所以"第二个视频客户端"实际是第三个，503 是正确行为。**压测前必须确认没有其它客户端连着。**

### 契约澄清

`on_upgrade` 拒绝时的**状态码是契约，响应体的成帧不是**：mongoose 原样发出 body，lws 用 `lws_return_http_status()` 包一层 HTML 页。两者都传达了消息，没有客户端依赖这些字节。已写进 `ws_transport.h`，测试断言改为检查消息子串而非逐字节相等。

---

## 已知会踩的坑（来自本仓库历史）

- **Docker 挂载缓存**：编辑过的文件在容器里可能是陈旧/截断的，报假的 CMake 语法错误。改动多时先做 md5 对照，必要时 `docker restart ubuntu_dev_x86`
- **禁止并发跑两个构建**：都会 `rm -rf build` 互删，报 `can't create ....o`
- **VPSS**：切换 app 前后确认 `dmesg` 无 `vpss_sbm_err` / `Oops`，open/release 应配对
