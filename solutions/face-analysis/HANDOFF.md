# face-analysis 生产交接

**最近大更新**: 2026-04-29
**作者**: Claude Opus 4.7 + 用户协作

---

## 一、生产配置（验证通过）

```
yolov8n-face INT8 (3MB)  →  FairFace INT8 (21MB)  +  HSEmotion BF16 (10MB)
                                 ↓
                          MQTT recamera/face-analysis/results
```

| 模块 | cvimodel | 功能 | ION |
|---|---|---|---|
| Face detect | `yolov8n_face_cv181x_int8.cvimodel` | bbox 检测 | ~6MB |
| Age/Gender/Race | `fairface_int8.cvimodel` | 7 race + 2 gender + 9 age bin | ~21MB |
| Emotion | `enet_b0_8_best_afew_cv181x_bf16.cvimodel` | 8-class FER+ | ~10MB |
| **总 ION** | | | **~37MB**（~60MB 池内）|

**实测性能**：detect 47-84ms，total 144-269ms（emotion 跳帧），1 face/frame，无 SEGV，age/gender/race/emotion 全部真分类（不再死锁）。

---

## 二、关键 commit（按时序）

```
d6f8eb6 chore: point sscma-micro submodule at our fork (suharvest/SSCMA-Micro)
0649780 Revert "chore: revert sscma-micro to upstream main"  ← 维持 fork patches
624116c chore: revert sscma-micro to upstream main           ← 失败实验：BF16 interleaved ION OOM
82462fe feat: switch defaults to yolov8n-face + FairFace AGR
65e61e8 feat: add PFLD landmark + ArcFace alignment for AGR  ← 默认关，opt-in
fb7572f chore: rebase sscma-micro onto upstream main with YoloV8 fixes
547bd62 chore: bump sscma-micro for YoloV8 num_class_ fix
f4e9c9b chore: bump sscma-micro for YoloV8 F32 output_box index fix
d29d0f5 docs(face-analysis): add HANDOFF.md (历史)
91cdb60 fix: use InsightFace normalization for AGR preprocessing
52dd7ae chore: bump sscma-micro to include Yolo11/V8 INT8 quant_param fix
f201c8f fix: guard face_detector against NaN / out-of-range bbox
3bdbe59 feat: face_detector model-agnostic on bbox xy convention
```

子模块（`suharvest/SSCMA-Micro` fork）：
- Branch `fix/yolov8-bugs` HEAD `cf928f2` — 3 个 yolov8 bug fix
- 上游 PR：https://github.com/Seeed-Studio/SSCMA-Micro/pull/99 (tensor indices), #100 (INT8 quant)

---

## 三、产物 md5

| 产物 | 路径 | md5 |
|---|---|---|
| **yolov8n int8** (生产用) | `model_conversion/recamera_yolo_face/yolov8n_face_derronqi/yolov8n_face_cv181x_int8.cvimodel` | `5a8b7de7...8f79d` (3.3MB) |
| **FairFace int8** (生产用) | `model_conversion/recamera_fairface/model_workspace/fairface_int8.cvimodel` | `65d2e630...96abe9` (21MB) |
| HSEmotion BF16 (生产用) | `/userdata/local/models/enet_b0_8_best_afew_cv181x_bf16.cvimodel` (在设备) | (~10MB) |
| **deb** (含 fork patches) | `solutions/face-analysis/build/face-analysis_0.1.1_riscv64.deb` | `c94e55a6...4f1c1` (170KB) |
| 设备 binary md5 | `/usr/local/bin/face-analysis` | `6b84100d...8f6a` (stripped) |

备选模型（不在生产路径，备查）：

| 备选 | md5 | 用途 |
|---|---|---|
| `yolov8n_face_cv181x_mixfp16.cvimodel` | `e196d1cf...39ad2c` | 同 yolov8n int8，仅 mixfp16 量化（cls heads BF16）|
| `yolov8n_face_cv181x_bf16_interleaved.cvimodel` | `ce482494...77f6e9` | 脱钩 fork 的尝试（BF16 interleaved），ION 47MB OOM 不可用 |
| `pfld_5point_bf16.cvimodel` | `0cebfa4b...167ee3` | 关键点检测（FairFace 不需，备 InsightFace 用）|

---

## 四、快速重新部署（设备上线后）

### 一键脚本
```bash
cd /Users/harvest/project/recamera/sscma-example-sg200x/solutions/face-analysis

# 推三个 cvimodel
sshpass -p recamera scp \
    /Users/harvest/project/recamera/model_conversion/recamera_yolo_face/yolov8n_face_derronqi/yolov8n_face_cv181x_int8.cvimodel \
    /Users/harvest/project/recamera/model_conversion/recamera_fairface/model_workspace/fairface_int8.cvimodel \
    recamera@192.168.42.1:/tmp/

# 推 deb + 装 + 切 conf + 启动
sshpass -p recamera ssh recamera@192.168.42.1 "
echo recamera | sudo -S /etc/init.d/S92face-analysis stop 2>&1 | tail -1
echo recamera | sudo -S killall -9 face-analysis 2>/dev/null
echo recamera | sudo -S /etc/init.d/S92facemesh-reader stop 2>&1 | tail -1
echo recamera | sudo -S mv /tmp/yolov8n_face_cv181x_int8.cvimodel /tmp/fairface_int8.cvimodel /userdata/local/models/
echo recamera | sudo -S rm -f /usr/local/bin/face-analysis
echo recamera | sudo -S opkg install --force-reinstall /tmp/face-analysis_0.1.1_riscv64.deb
echo recamera | sudo -S md5sum /usr/local/bin/face-analysis  # 应该 6b84100d...
echo recamera | sudo -S sed -i 's|^FACE_MODEL=.*|FACE_MODEL=/userdata/local/models/yolov8n_face_cv181x_int8.cvimodel|' /etc/face-analysis.conf
echo recamera | sudo -S sed -i 's|^GENDERAGE_MODEL=.*|GENDERAGE_MODEL=/userdata/local/models/fairface_int8.cvimodel|' /etc/face-analysis.conf
echo recamera | sudo -S rm -f /var/log/face-analysis.log
echo recamera | sudo -S /etc/init.d/S92face-analysis start
"
```

### 验证
```bash
# 等首帧 + 看 MQTT
sshpass -p recamera ssh recamera@192.168.42.1 "timeout 18 mosquitto_sub -h localhost -t 'recamera/face-analysis/results'"
```

---

## 五、reCamera 部署核心 gotcha 速查

详见 memory `recamera-deploy-gotchas.md`。最坑的几条：

1. **每次 BF16 / OOM 实验后必须物理断电**（不是 `reboot`）— kernel VPSS Grp(0) 状态污染软重启清不掉
2. **CPack strip 步骤产生不同 md5** — `build/face-analysis` (unstripped) ≠ deb 内 binary (stripped)，调试看 deb 时要 `ar x + tar xzf` 验
3. **opkg install --force-reinstall 不一定真换 binary** — 验证 md5 才靠谱，必要时 `rm -f /usr/local/bin/face-analysis` 后再装
4. **多次 stop/start 累积 VPSS 状态污染** — 同一启动周期超过 5-10 次 stop/start 几乎必崩，需断电
5. **ION 池约 60MB 上限** — yolo BF16 (16MB) + FairFace (21MB) + emotion BF16 (10MB) ≈ 47MB 触 OOM；INT8 yolo (6MB) + FairFace + emotion ≈ 37MB OK
6. **`--quantize INT8` ≠ `--quant_output`** — 内部 INT8 不等于输出 INT8，输出仍是 F32（默认），detector 走 F32 path
7. **opkg install 重置 conf 到 deb 默认** — 切换 conf 顺序：先 install，再 sed 改 conf
8. **opkg install 留 .DS_Store warning** — 无害但 deb 不该带 macOS 元数据，可在 build/ 加 `.gitattributes` 屏蔽

---

## 六、待续工作（下次接手）

### 6.1 上游 PR 跟进
- PR1 #99: YoloV8 tensor indices (F32 output_box + num_class_)
- PR2 #100: INT8 quant_param swap (YoloV8 + Yolo11)

合并后：
1. submodule URL 切回 `Seeed-Studio/SSCMA-Micro`
2. submodule HEAD 跟随 upstream main
3. 验证生产 deb（应当无变化，因为 fork branch 跟 upstream 内容应一致）

### 6.2 优化点（不阻塞）
- **新增模型时先看 ION 影响**：参考 memory `cv181x-ion-budget.md`
- **未真正脱钩 fork**：BF16 yolo interleaved 验证可行但 ION OOM。若上游 PR 合并 → 自然脱钩，不需要 BF16 路径
- **PFLD landmark 代码 dead but kept**：留作以后 InsightFace 备用，不删

### 6.3 已知 known limitation
- **age 是 9 个 bin** （0-2 / 3-9 / 10-19 / ... / 70+），不是连续年龄
- **InsightFace genderage 永远不要再用**（见 memory `insightface-genderage-broken.md`）
- 偶尔 emotion confidence 0.00（cached frame，跳帧策略），正常

---

## 七、相关 memory 索引

```
~/.claude/projects/-Users-harvest-project-recamera-sscma-example-sg200x/memory/MEMORY.md
```

关键今天写的：
- `insightface-genderage-broken.md` — InsightFace 模型死锁 mean
- `face-analysis-fairface-replacement.md` — FairFace 替代方案
- `sscma-micro-yolov8-f32-bug.md` — F32 path output_box typo
- `cvi-tpu-output-defaults-f32.md` — `--quantize` vs `--quant_output` 区分
- `tpu-mlir-calibration-oom-mac.md` — Mac docker 7.7GB ION 限制
- `yolov11n-face-model-quality-cap.md` — yolov11n-face 模型本身弱
- `fleet-sudo-and-wsl-docker-proxy.md` — fleet --sudo + WSL docker 网络
- `recamera-deploy-gotchas.md` — reCamera 部署坑（**新写**）
- `cv181x-ion-budget.md` — ION 预算 + 模型组合参考（**新写**）

---

**结束**。下次接手从"四、快速重新部署"开始。
