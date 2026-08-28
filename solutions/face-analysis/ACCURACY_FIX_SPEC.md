# face-analysis（一代 / cv181x）准确度修复 spec

对应 reCamera Pro 侧 0.2.0 的同类修复。**一代只需要改 3 条**，Pro 的另外两条不适用（见末尾"不要做的事"）。

设计依据：`recamera_pro/kit/logic/tracker.py`、`recamera_pro/kit/logic/attributes.py`、`recamera_pro/docs/guide/face-attribute-accuracy.md`。语义要和 Pro 侧一致，实现用 C++ 重写（**不要**试图复用 Python）。

---

## 一、要修的三条

### F1. `face.id` 不是身份，是个递增计数器

`main/face_detector.cpp:143`

```cpp
face.id = face_id_counter_++;
```

`face_id_counter_` 在构造函数里初始化为 0（`main/face_detector.cpp:17`），此后只增不重置、不做任何跨帧关联。**每一帧的每一张脸都拿到全新编号。**

而 `main/face_detector.h:18` 的注释写的是 `int id;  // Face ID for tracking`，`main/mqtt_payload.cpp` 又把它原样发布成 JSON 的 `"id"` 字段。任何消费方按字面理解都会当身份用。

**修法**：引入真正的 IoU 跟踪器，`face.id` 改为承载稳定的 track id。

### F2. 属性是单帧 argmax，没有时序平滑

同一个人的 race / age 逐帧跳变。

**修法**：按 track id 累积各 head 的**概率向量**，报告累积和的 argmax。

### F3. 没有质量门控

`main/face_detector.cpp:139` 只有 `if (face.w < 0.01f || face.h < 0.01f) continue;` —— 那是防止裁剪崩溃的保护（画面 1%），不是属性质量门。30px 的脸放大到分类器输入后基本是插值伪影，分类头照样给出看起来很自信的标签。

**修法**：按**原图像素**的人脸框短边过滤，低于阈值的脸不跑 stage 2/3。

---

## 二、实现

### 2.1 新增 `main/face_tracker.{h,cpp}`

```cpp
namespace face_analysis {

struct FaceTrackerConfig {
    float iou_threshold   = 0.3f;   // 匹配阈值
    int   max_lost_frames = 15;     // 连续丢失多少帧后丢弃身份
};

class FaceTracker {
public:
    explicit FaceTracker(const FaceTrackerConfig& cfg = {});

    // 输入本帧检测（归一化坐标），返回与 faces **下标对齐**的 track id 数组。
    // 每个检测要么关联到已有 track，要么新建一个。
    std::vector<int> update(const std::vector<FaceInfo>& faces);

    // 本次 update 中被淘汰的 track id（供证据表回收内存）
    const std::vector<int>& removedIds() const;

    int trackFrames(int track_id) const;   // 该 track 已被连续跟踪的帧数
};

}  // namespace face_analysis
```

**匹配策略**（贪心，够用，不要引入卡尔曼）：

1. 已有 track 按 `frames_tracked` 降序（老 track 优先争夺检测）
2. 对每个 track，在未被占用的检测里找 IoU 最大且 > `iou_threshold` 的
3. 未匹配上的 track `lost_frames++`，超过 `max_lost_frames` 则淘汰
4. 未匹配上的检测新建 track，分配单调递增 id

坐标用归一化 [0,1] 做 IoU（`FaceInfo` 本来就是归一化的），这样不同分辨率下阈值行为一致。

**不要**引入速度预测 / 边缘感知淘汰——Pro 侧有那些是因为它复用的是人体跟踪器，一代不需要。

### 2.2 新增 `main/attribute_evidence.{h,cpp}`

```cpp
namespace face_analysis {

struct EvidenceConfig {
    float min_face_px      = 64.f;  // 质量门控：原图人脸框短边（像素）
    int   min_track_frames = 3;     // 结论标记 stable 所需的通过门控帧数
    float decay            = 1.0f;  // 每帧对已累积证据的衰减，1.0 = 等权求和
};

// 质量门控。face 是归一化坐标，fw/fh 是原图尺寸。
bool passesGate(const FaceInfo& face, int fw, int fh, const EvidenceConfig& cfg);

struct Verdict {
    int   index      = -1;
    float confidence = 0.f;   // 获胜标签的**票份额**（累积概率质量占比）
    int   frames     = 0;
    bool  stable     = false;
};

class AttributeEvidence {
public:
    explicit AttributeEvidence(const EvidenceConfig& cfg = {});

    // head 用固定枚举，不要用字符串 key（嵌入式，别做 map<string,...> 的哈希）
    enum Head { HEAD_RACE = 0, HEAD_GENDER, HEAD_AGE, HEAD_EMOTION, HEAD_COUNT };

    void    add(int track_id, Head head, const float* probs, int n);
    void    bumpFrame(int track_id);          // 每帧每 track 调一次，在所有 add 之后
    Verdict verdict(int track_id, Head head) const;
    void    sweep(const std::vector<int>& removed_ids);   // 回收已淘汰 track

    void    setConfig(const EvidenceConfig& cfg);
};

}  // namespace face_analysis
```

**语义必须与 Pro 侧一致**（见 `kit/logic/attributes.py`）：

- `add` 做的是 `sums[head] = sums[head] * decay + probs`
- `verdict` 返回累积和的 argmax，`confidence = 累积和[argmax] / 累积和总和`（**票份额，不是单帧 softmax**）
- `stable` 当且仅当 `frames >= min_track_frames`
- `passesGate` 用 `min(w*fw, h*fh) < min_face_px` 判定

**内存**：每个 track 存 7+2+9+8 = 26 个 float，用定长数组，不要 `std::map<std::string, std::vector<float>>`。track 表用 `std::vector` 或小的 `std::unordered_map<int, Evidence>` 都行，但必须在 `sweep` 里真正释放。

### 2.3 扩展 `AgeGenderRaceResult` 暴露概率向量

`main/age_gender_race_runner.h:12` 的 `AgeGenderRaceResult` 目前只有 argmax + score，**没有概率向量**，而 F2 必须拿到概率向量才能累积。

在结构体里加：

```cpp
float race_probs[7]   = {};
float gender_probs[2] = {};
float age_probs[9]    = {};   // InsightFace 路径下这些保持全 0
```

在 runner 内部**已经做 softmax 的地方**顺手填充（不要重算一遍 softmax）。InsightFace 分支（`is_fairface == false`）不填，保持全 0——调用方靠 `is_fairface` 分流。

情绪那边已经有 `FaceAttributes::emotion_probs`（`main/attribute_analyzer.h:83`），直接用。

### 2.4 改 `AttributeAnalyzer`

`main/attribute_analyzer.cpp:138` 的 `analyzeAll`：

1. 签名加一个 track id 数组参数（与 `faces` 下标对齐），或者让 `FaceInfo::id` 在进来之前就已经是 track id —— **选后者**，改动面更小：`main.cpp` 在调 `analyzeAll` 之前先跑 tracker 把 `faces[i].id` 覆写成 track id。
2. 循环体开头加门控：`if (!passesGate(face, fw, fh, cfg)) { /* 属性全部留空，标记 gated */ results.push_back(analyzed); continue; }` —— **在任何裁剪和推理之前**，被门控的脸不消耗推理。
3. AGR 推理成功后，把三个概率向量 `add()` 进证据表；情绪推理成功后 `add()` 情绪概率。
4. 该帧该 track 的所有 `add` 完成后调一次 `bumpFrame`。
5. 写回 `analyzed.attributes` 的**不再是单帧结果，而是 `verdict()` 的结果**：label 由 verdict 的 index 查表得到，confidence 用 verdict 的票份额。
6. `FaceAttributes` 加两个字段：`bool gated = false; bool stable = false; int evidence_frames = 0;`

### 2.5 删掉 IoU 情绪缓存

`main/attribute_analyzer.cpp:253-305` 那段"按 IoU>0.3 匹配上一帧 bbox 复用情绪"的逻辑，连同 `EmotionCache` / `last_emotion_`（`main/attribute_analyzer.h:126-133`）**整体删除**。

有了真跟踪 + 证据累积之后它是多余的：不跑情绪的帧上，`verdict(track_id, HEAD_EMOTION)` 返回的本来就是该 track 自己的累积结论。而且它比 IoU 匹配更准（IoU 匹配在两张脸靠近时会错配）。

`emotion_interval_` 的跳帧节奏**保留不变**。

### 2.6 `main.cpp` 接线与配置

- 全局加 `FaceTracker* g_face_tracker`，在 `analyzeAll`（`main/main.cpp:522`）之前跑 `update()` 并把 track id 覆写进 `faces[i].id`
- 每帧 `sweep(g_face_tracker->removedIds())`
- 新增 CLI 选项（照 `main/main.cpp:112-127` 的 `long_options` 风格加，注意 option 编号接着 7 往下排）：
  - `--min-face-px N`（默认 64）
  - `--min-track-frames N`（默认 3）
  - `--track-max-lost N`（默认 15）
  - `--evidence-decay F`（默认 1.0）
- `print_usage` 里同步加说明

### 2.7 `mqtt_payload.cpp` 输出

- `"id"` 现在是真 track id，**在 `main/mqtt_payload.h` 或 JSON 里注明语义变了**
- 新增 `"track_id"`（与 `id` 同值，语义明确的别名，方便消费方迁移）、`"gated"`、`"stable"`、`"evidence_frames"`
- `*_confidence` 的语义变成票份额 —— 在 `main/mqtt_payload.cpp` 顶部加注释说明

---

## 三、不要做的事

- **不要**加客群直方图 / demographics 聚合。一代根本没有这个功能，Pro 侧那条修的是 Pro 自己新加错的东西。加功能不在本次范围。
- **不要**动 blur / privacy 相关代码。一代没有 top-K 切片，Pro 那条不适用。`--max-regions` 的 8 区域上限是另一个问题，本次不碰。
- **不要**动 landmark 对齐路径（`main/attribute_analyzer.cpp:180-215`）。它已经实现好了，只是默认没加载模型。启不启用要等离线评测数字，不在本次范围。
- **不要**改任何模型文件、`.cvimodel`、模型路径默认值。
- **不要**动 `/etc/init.d/S93sscma-supervisor` 的启用状态（可以 stop，不要改名/删除）。

---

## 四、验收

1. Docker 容器内编译通过，`cpack` 出 deb
2. 自己写一个**不依赖硬件**的单元测试或独立 main，验证：
   - `FaceTracker`：同一张脸连续多帧 id 不变；两张脸交换检测顺序后 id 不跟随下标；脸消失超过 `max_lost_frames` 后 id 被回收
   - `AttributeEvidence`：多数帧投 A、少数帧投 B 时 verdict 为 A；`confidence` 是票份额（可手算核对）；`stable` 在 `min_track_frames` 之后才为 true；`sweep` 后该 track 的证据消失
   - `passesGate`：边界值（正好等于 `min_face_px` 应通过）
3. 交叉编译产物 md5


---

## 五、设备实测（实机可用）

设备：`recamera@192.168.42.1`，密码由派发方在环境变量 `RECAMERA_PASS` 中给出（**不要把密码写进任何文件或提交**）。

### 5.1 实测前的设备现状（已由主线程确认，不用重查）

```
Linux reCamera 5.10.4 riscv64,  /userdata 剩余 3.4G
/etc/init.d:  K03node-red  K91sscma-node  K92face-analysis
              K92facemesh-reader  S93sscma-supervisor   <-- 唯一启用的相机冲突服务
当前无 face-analysis / node-red / sscma 进程在跑
/usr/local/bin/face-analysis 已安装（2271344 bytes, Aug 8 2026）
```

`/userdata/local/models/`（需 sudo 才能 ls）已有：
- `yolov8n_face_cv181x_int8.cvimodel` (3.3M) — conf 的 `FACE_MODEL`
- `fairface_int8.cvimodel` (21M) — conf 的 `GENDERAGE_MODEL`
- **缺** `enet_b0_8_best_afew_cv181x_bf16.cvimodel` — conf 的 `EMOTION_MODEL` 指向它但设备上没有

本地有这个文件，路径：
`model_conversion/recamera_emotion/model_workspace/enet_b0_8_best_afew_cv181x_bf16.cvimodel`
**部署时一并推上去**，否则情绪那一级不会初始化，F2 的情绪投票无法验证。

ION 预算：yolo int8 (~6M) + FairFace (~21M) + emotion bf16 (~10M) ≈ 37M，上限约 60M。本次不新增模型，预算不变。

### 5.2 部署步骤

参考 `solutions/ppocr-reader/deploy.sh` 的模式。顺序**必须**是：

1. 停冲突服务：`/etc/init.d/S93sscma-supervisor stop`，再 `killall -9 face-analysis`，**等 2-3 秒**
2. 推 deb 和缺的情绪模型到 `/tmp/`
3. `sudo mv` 情绪模型进 `/userdata/local/models/`
4. `sudo rm -f /usr/local/bin/face-analysis` 后再 `opkg install --force-reinstall`
   （HANDOFF 踩坑 #3：`--force-reinstall` 不一定真换 binary，**必须 md5 验证**）
5. **`opkg install` 会把 conf 重置成 deb 默认**（HANDOFF 踩坑 #7），所以改 conf 必须在 install **之后**
6. `/etc/init.d/K92face-analysis start`

### 5.3 必须遵守的设备 gotcha（HANDOFF 五节，逐条）

- **同一启动周期内 stop/start 超过 5-10 次几乎必崩**（VPSS 状态污染累积）。控制迭代次数；如果 VPSS 持续失败，**停下来报告，让人来断电**——`reboot` 清不掉，必须物理断电
- 每次改 binary 后**必须 md5 验证**，别信 opkg 的输出
- CPack 的 strip 步骤让 `build/face-analysis` 和 deb 内 binary md5 不同，比对时注意比的是哪个

### 5.4 验收（每条都要原始输出，不要转述）

**V1 — 跟踪 id 稳定**：人脸在画面里连续停留，订阅 MQTT，断言同一张脸的 `"id"` 跨帧不变（旧实现每帧都变）。
```
mosquitto_sub -h localhost -t 'recamera/face-analysis/results' -C 30
```
贴出连续 30 条里同一张脸的 id 序列。**这是 F1 的直接证据。**

**V2 — 属性不再逐帧跳变**：同一段静止人脸，统计 `race` / `age_label` 在 30 帧内的取值分布。修好后应当是单一取值（或极少数变化）。**与修复前对照**——修复前的数据可以先用当前设备上的旧 binary 采一份再装新的。

**V3 — 票份额**：`*_confidence` 应随 `evidence_frames` 增长而收敛，且不再等于单帧 softmax 的值。

**V4 — 质量门控**：人退到远处让脸变小，断言出现 `"gated": true` 且属性字段为空/默认，同时该帧不产生 AGR 推理（看 verbose 日志的推理耗时下降）。

**V5 — 无崩溃**：跑满 3 分钟，`dmesg | grep -iE 'vpss|Oops|fail|error'` 干净，进程还活着。

### 5.5 禁止

- 禁止 `rm -rf` 非 build 产物目录
- 禁止 `mv`/`rename` 任何 `/etc/init.d/S*`（可以 `stop`，不可改名）
- 禁止 `sudo rm` 除 `/usr/local/bin/face-analysis` 以外的任何路径
- 禁止删除或覆盖 `/userdata/local/models/` 下已有的任何 cvimodel（只能新增缺的那个）
- 禁止改 `/etc/face-analysis.conf` 里的模型路径
- 遇到 VPSS 崩溃 / kernel Oops：**立即 STOP 并报告**，不要反复重启硬试

### 5.6 构建

只能用项目 CLAUDE.md 记录的 Docker 路径，禁止在 macOS 本机裸调 cmake：

```bash
docker start ubuntu_dev_x86
docker exec ubuntu_dev_x86 bash -c "
export SG200X_SDK_PATH=/workspace/sg2002_recamera_emmc
export PATH=/workspace/host-tools/gcc/riscv64-linux-musl-x86_64/bin:\$PATH
cd /workspace/sscma-example-sg200x/solutions/face-analysis
rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release . && cmake --build build -j4 && cd build && cpack
"
```

新增的 `.cpp` 会被 `main/CMakeLists.txt` 的 `file(GLOB_RECURSE ...)` 自动收，**通常不需要改 CMakeLists**——如果需要改，在报告里说明为什么。
