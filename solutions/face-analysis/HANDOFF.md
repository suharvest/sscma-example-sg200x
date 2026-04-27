# face-analysis 交接文档

**生成时间**: 2026-04-27
**作者**: Claude Opus 4.7 + 用户协作  
**目标读者**: 明天接手验证的 agent / 工程师

---

## 一、本次工作概要

修了 face-analysis 三个独立 bug + 一个加速优化，**所有代码改动已 commit**，但**没在设备上端到端验证**（设备 VPSS 反复挂掉，需要物理重启后跑一次干净流程即可）。

待办本质：**装 deb + 改一行 conf + 重启服务 + 抓 MQTT 数据**。

---

## 二、本次的 commit 列表（按依赖顺序）

```
91cdb60 fix(face-analysis): use InsightFace normalization for AGR preprocessing
52dd7ae chore: bump sscma-micro to include Yolo11/V8 INT8 quant_param fix
f201c8f fix(face-analysis): guard face_detector against NaN / out-of-range bbox
7d1a112 fix(face-analysis): include facemesh-reader in conflict-stop list + better deploy diagnostics
3bdbe59 feat(face-analysis): make face_detector model-agnostic on bbox xy convention
6a7f25b feat(face-analysis): add emotion frame-skip with IoU cache
21b115e feat(face-analysis): switch emotion model to HSEmotion enet_b0_8
6553412 chore: bump sscma-micro to re-add YoloSingle support
```

子模块 `components/sscma-micro/sscma-micro` 也有两个本地 commit（detached HEAD）：
```
34d2b6f fix(yolo): swap quant_param indices in Yolo11/V8 INT8 postprocess
034c350 feat: re-add YoloSingle support for single-output YOLO face models
```

---

## 三、预编译产物（明天直接用）

### 1. face-analysis deb（含全部 commit）
```
路径:  solutions/face-analysis/build/face-analysis_0.1.1_riscv64.deb
md5:   8de786dc47b9a5f8a4a4916b7734050b
大小:  ~155 KB
```

如果 build 目录被清掉了，重新编译：
```bash
docker exec ubuntu_dev_x86 bash -c "
export SG200X_SDK_PATH=/workspace/sg2002_recamera_emmc
export PATH=/workspace/host-tools/gcc/riscv64-linux-musl-x86_64/bin:\$PATH
cd /workspace/sscma-example-sg200x/solutions/face-analysis
cmake --build build -j4 && cd build && cpack
"
```

### 2. yolo11n-face cvimodel（已转，未推到设备或推过了被擦掉）

| 文件 | 大小 | md5 | 用途 |
|---|---|---|---|
| `model_conversion/recamera_yolo_face/model_workspace/yolov11n_face_cv181x_int8.cvimodel` | 2.9 MB | `1ba9db9c88425a32af84fc816fcc3063` | 主选 — 速度优先 |
| `model_conversion/recamera_yolo_face/model_workspace/yolov11n_face_cv181x_mixfp16.cvimodel` | 3.3 MB | `3c111e6270ac496ac48594d62421f89e` | 备选 — 精度优先（11 层 BF16） |
| `model_conversion/recamera_yolo_face/model_workspace/yolov11n_face_cv181x_bf16.cvimodel` | 6.7 MB | `905204be7da9f3fd108f1c93b13b27d9` | 不要用 — ION OOM |

### 3. emotion BF16（生产模型，已在设备上）

```
设备路径:  /userdata/local/models/enet_b0_8_best_afew_cv181x_bf16.cvimodel
本地路径:  model_conversion/recamera_emotion/model_workspace/enet_b0_8_best_afew_cv181x_bf16.cvimodel
大小:      9.9 MB
```

---

## 四、明天的验证流程（按顺序）

### 准备：物理重启设备 + 等 SSH 起来
```bash
until sshpass -p recamera ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 \
    recamera@192.168.42.1 "echo OK" 2>/dev/null | grep -q OK; do sleep 5; done
sshpass -p recamera ssh recamera@192.168.42.1 "uptime"
```

⚠️ 如果之前 SSH host key 不匹配：`ssh-keygen -R 192.168.42.1` 重置一下。

### 步骤 1: 推 INT8 yolo11n-face 模型 + 安装新 deb（一次性）

```bash
# 推模型
sshpass -p recamera scp \
    /Users/harvest/project/recamera/model_conversion/recamera_yolo_face/model_workspace/yolov11n_face_cv181x_int8.cvimodel \
    recamera@192.168.42.1:/tmp/

# 推 deb
sshpass -p recamera scp \
    /Users/harvest/project/recamera/sscma-example-sg200x/solutions/face-analysis/build/face-analysis_0.1.1_riscv64.deb \
    recamera@192.168.42.1:/tmp/mine.deb

# 装 + 切默认 conf 到新 yolo + 重启服务
sshpass -p recamera ssh recamera@192.168.42.1 "
    echo recamera | sudo -S /etc/init.d/S92face-analysis stop
    echo recamera | sudo -S /etc/init.d/S92facemesh-reader stop
    echo recamera | sudo -S mv /tmp/yolov11n_face_cv181x_int8.cvimodel /userdata/local/models/
    echo recamera | sudo -S opkg install --force-reinstall /tmp/mine.deb
    echo recamera | sudo -S sed -i 's|^FACE_MODEL=.*|FACE_MODEL=/userdata/local/models/yolov11n_face_cv181x_int8.cvimodel|' /etc/face-analysis.conf
    sleep 3
    echo recamera | sudo -S rm -f /var/log/face-analysis.log
    echo recamera | sudo -S /etc/init.d/S92face-analysis start
    sleep 12
    ps -A 2>/dev/null | grep face-analysis | grep -v grep | head -1
"
```

**期望结果**：
- 进程活着（PID 显示）
- `dmesg | grep signal` 应该是空的（即没有新 SEGV）

### 步骤 2: 抓 MQTT 验证（脸对准摄像头）

```bash
timeout 30 mosquitto_sub -h 192.168.42.1 -t 'recamera/face-analysis/results' 2>&1 | python3 -c "
import json, sys
samples = []
for line in sys.stdin:
    try:
        d = json.loads(line)
        n = d.get('face_count', 0)
        ms = d.get('inference_time_ms')
        if n > 0:
            f = d['faces'][0]
            samples.append((d['frame_id'], ms, n, f.get('age'), f.get('gender'), f.get('emotion'), f.get('emotion_confidence')))
            if len(samples) >= 12: break
    except: pass

for fid, ms, fc, age, gender, emotion, em_conf in samples:
    print(f'frame={fid} ms={ms} face_count={fc} age={age} gender={gender} emotion={emotion}({em_conf:.2f})')
"
```

### 步骤 3: 三个检查项 + 决策树

#### Check A: face_count = 1（验证 Yolo11 quant fix）
- ✅ **face_count = 1** （或 0 没人脸时）→ Yolo11 quant fix 生效，YoloSingle 依赖可移除
- ❌ face_count > 1（10 个重复框）→ Yolo11 quant fix **没起作用**。回退步骤：
  ```bash
  # 改回老 yolo-face
  sshpass -p recamera ssh recamera@192.168.42.1 "
      echo recamera | sudo -S sed -i 's|^FACE_MODEL=.*|FACE_MODEL=/userdata/local/models/yolo-face_mixfp16.cvimodel|' /etc/face-analysis.conf
      echo recamera | sudo -S /etc/init.d/S92face-analysis restart
  "
  ```
  然后让另一个 codex agent 检查 `components/sscma-micro/sscma-micro/sscma/core/model/ma_model_yolo11.cpp` 的 L129/L136 是否真的修了对（应该 score 用 `outputs_[i*2+1].quant_param`，DFL box 用 `outputs_[i*2].quant_param`）。

#### Check B: 推理时间 ~50-200ms（验证 Yolo11 SEGV/perf fix）
- ✅ 0-face 帧 ~54ms，1-face 帧 80-200ms（emotion BF16 主导）→ 合理
- ❌ 1-face 帧 > 1000ms 或 dmesg 有 SEGV → patch 没生效
  - 查 `dmesg | grep signal`
  - 查 `/var/log/face-analysis.log` 末尾
  - 如果 SEGV 还在，**先回退到老 yolo-face**（同 Check A 失败的回退）

#### Check C: age 不再固定 35（验证 AGR preprocessing fix）
- ✅ 不同人不同 age（同一人不同帧也应该 ±2 岁波动）→ AGR 预处理修复生效
- ❌ 仍然 age=35 不变 → 进一步排查（concrete next test in `task ae95c59adb2d9b4ec.output` codex spec）：在 `age_gender_race_runner.cpp:480` 后插一行 `MA_LOGI` 打印 `vals[2]` 原始值，看模型输出本身是否仍卡在 0.35

### 步骤 4（可选）：Mixfp16 yolo 对比

如果 INT8 yolo 检测精度感觉不够（小脸漏检/远距离偏差），可以切到 mixfp16 看是否更准：

```bash
sshpass -p recamera scp \
    /Users/harvest/project/recamera/model_conversion/recamera_yolo_face/model_workspace/yolov11n_face_cv181x_mixfp16.cvimodel \
    recamera@192.168.42.1:/tmp/
sshpass -p recamera ssh recamera@192.168.42.1 "
    echo recamera | sudo -S /etc/init.d/S92face-analysis stop
    echo recamera | sudo -S mv /tmp/yolov11n_face_cv181x_mixfp16.cvimodel /userdata/local/models/
    echo recamera | sudo -S sed -i 's|^FACE_MODEL=.*|FACE_MODEL=/userdata/local/models/yolov11n_face_cv181x_mixfp16.cvimodel|' /etc/face-analysis.conf
    echo recamera | sudo -S /etc/init.d/S92face-analysis start
"
```

A/B 比较推理时间 + 检测稳定性，记入 memory 便于后续决策。

---

## 五、设备已知坑 — 严守

### 5.1 一次切换原则
**不要反复 stop/start face-analysis**。每次 stop/start 都在 VPSS 状态里累积污染，10+ 次几乎必然挂掉（dmesg 出现 `vpss_get_chn_frame: get chn frame fail` → 进程活但 0 帧）。
- 想 A/B 测：建议**两次完整重启之间各做一次部署**，而不是同一 boot 周期切换 5 次
- 详细见 `~/project/app_collaboration/.claude/skills/solution-validation/references/gotchas-recamera.md` 第 3 条

### 5.2 facemesh-reader 抢摄像头
boot 时 `S92facemesh-reader` 会自动起来，先于 face-analysis 抢 VPSS Grp(0)。
- 修复已在 init 脚本 `S92face-analysis::stop_conflicting_services` 里加了 `facemesh-reader`，但**第一次 start 还是要手动先 stop facemesh-reader**（boot 时它已在跑）
- deploy.sh 也已更新加上这条

### 5.3 SSH host key 频繁变
```bash
ssh-keygen -R 192.168.42.1
# 然后第一次连接时加 -o StrictHostKeyChecking=accept-new
```

---

## 六、待 commit 但未做的（如果验证通过）

如果 Check A + B + C 全部通过（INT8 yolo11n-face 真的工作 + AGR age 修了）：

### 6.1 切默认值
工作区目前 main.cpp 默认 face_model 是老的 `yolo-face_mixfp16.cvimodel`。验证通过后改成新模型默认：
```cpp
// solutions/face-analysis/main/main.cpp:26
std::string face_model = "/userdata/local/models/yolov11n_face_cv181x_int8.cvimodel";
```
+ `solutions/face-analysis/rootfs/etc/face-analysis.conf` 也改 `FACE_MODEL=...int8.cvimodel`

commit message 用："feat(face-analysis): switch default to yolov11n-face INT8 (multi-output)"

### 6.2 删 YoloSingle 类（可选清理）
切默认到多输出 yolo11n 后，YoloSingle 在 face-analysis 里彻底用不到了。可以从 sscma-micro 子模块 fork 里删掉本地的 `034c350` commit：
```bash
cd components/sscma-micro/sscma-micro
git rebase -i 60e2ede  # drop commit 034c350, keep 34d2b6f
# 然后父仓库 git add submodule + commit "chore: drop YoloSingle now that multi-output yolo works"
```
但**老 yolo-face_mixfp16 cvimodel 仍在设备上做兜底**，删 YoloSingle 后该模型就跑不起来了——评估完风险再删，不急。

### 6.3 提 PR 给 Seeed-Studio/SSCMA-Micro
sscma-micro fork 里的 `34d2b6f` Yolo11/V8 INT8 quant fix 是真 bug，**值得提到上游**：
- 仓库：https://github.com/Seeed-Studio/SSCMA-Micro
- PR 内容就是 commit 34d2b6f 的 diff（2 个文件，~7 行）
- PR 描述参考 commit message
- 期待响应：Seeed 团队对自家 reCamera 投入大，应该会接

YoloSingle commit `034c350` 不用提（fork 私货，用户态多输出代替即可）

---

## 七、其它已就绪的待办（不阻塞验证）

### 7.1 内核 RGN MOSAIC patch（路径 A，5 行）
- 文件：`sg2002_recamera_emmc/osdrv/0001-rgn-vpss-improve-MOSAIC-privacy-mask-visibility-on-CV.patch`
- 内核 patch，把硬件马赛克从随机噪声变成半透明纯色块
- 用不到，因为 `BLUR_ENABLED=0` 默认关。哪天想开就 apply + 重编 cv181x_rgn.ko / cv181x_vpss.ko
- 详见 memory `cvi-rgn-mosaic-patch.md`

### 7.2 用户态真马赛克设计（路径 B）
- 250-370 行的 OVERLAYEX 设计
- 设计文档存在 memory `face-blur-overlayex-design.md`
- 任何时候想做真马赛克都可以照做

### 7.3 Mixed-precision INT8 emotion
- 半成品（agent 跑了 22 分钟才扫 27 层就被 kill 了）
- 结论：**不值得做**——SE block 量化必须保 BF16，dtype boundary 开销吃掉 INT8 的速度优势，且精度收回去不到 BF16 水平
- 详见 memory `int8-efficientnet-cv181x-not-worth.md`

---

## 八、快速 sanity check 清单（首次验证完后回头看）

- [ ] face_count == 1 for 1 face in frame
- [ ] inference_time_ms < 250ms for 1-face frames
- [ ] age 不再固定 35（同一人不同帧 ±2 年抖动算正常）
- [ ] gender 与脸对应（不再无意义乱跳）
- [ ] emotion 字段是 8 类 FER+（angry/contempt/disgust/fear/happy/neutral/sad/surprise 之一）
- [ ] emotion_probs JSON 含 8 个 key
- [ ] dmesg 无 face-analysis SEGV
- [ ] dmesg 无 vpss_get_chn_frame fail
- [ ] 服务起来后 30 秒持续发布 MQTT，不间断
- [ ] EMOTION_INTERVAL=2 生效（看 inference_time_ms 一帧 ~200ms 一帧 ~60ms 交替）

---

## 九、紧急回退路径

如果**任何环节炸了，要紧急让设备恢复生产能用状态**：

```bash
sshpass -p recamera ssh recamera@192.168.42.1 "
    echo recamera | sudo -S /etc/init.d/S92face-analysis stop
    echo recamera | sudo -S sed -i 's|^FACE_MODEL=.*|FACE_MODEL=/userdata/local/models/yolo-face_mixfp16.cvimodel|' /etc/face-analysis.conf
    echo recamera | sudo -S sed -i 's|^EMOTION_MODEL=.*|EMOTION_MODEL=/userdata/local/models/enet_b0_8_best_afew_cv181x_bf16.cvimodel|' /etc/face-analysis.conf
    echo recamera | sudo -S /etc/init.d/S92face-analysis start
"
```

老 `yolo-face_mixfp16` 一直在设备上没删，搭配 BF16 emotion + AGR 修复（明天 deb 自带），最差也能跑回原状态。

---

## 十、相关文件速查

```
代码:
  solutions/face-analysis/main/                 应用代码
  components/sscma-micro/sscma-micro/sscma/     上游+本地补丁
  solutions/face-analysis/build/*.deb           编译产物

模型:
  model_conversion/recamera_yolo_face/          yolo11n-face 转换项目（INT8/BF16/mixfp16 都已转好）
  model_conversion/recamera_emotion/            HSEmotion 转换项目（BF16 在用，INT8/mix 已废弃）

文档:
  HANDOFF.md (本文件)                          交接
  ~/.claude/projects/.../memory/MEMORY.md      项目 memory 索引
  ~/project/app_collaboration/.claude/skills/solution-validation/references/gotchas-recamera.md  reCamera 通用坑

子模块 commit (sscma-micro fork):
  034c350  YoloSingle (re-add) — 切到 yolo11n 后可删
  34d2b6f  Yolo11/V8 INT8 quant fix — 真 bug, 提 PR

内核 patch (待 apply):
  sg2002_recamera_emmc/osdrv/0001-rgn-vpss-improve-MOSAIC-*.patch  blur 视觉优化
```

---

**结束。明天重启设备后从"四、明天的验证流程"开始往下做。**
