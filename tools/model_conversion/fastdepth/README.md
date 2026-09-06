# FastDepth → cvimodel 转换

FastDepth (`dwofk/fast-depth`, `mobilenet-nnconv5dw-skipadd-pruned` variant)
monocular depth estimation, converted to an INT8 `cvimodel` for reCamera
(CV181x). 转换出的产物随 `solutions/depth-estimation` 一起分发，见
`solutions/depth-estimation/model/`。本目录只放**转换代码**，中间产物
（ONNX、mlir、npz、校准表、校准图）都不进 git。

- **默认发布 BF16**（`fastdepth_224_bf16.cvimodel`），INT8 作为进阶选项
- Input: `1x3x224x224`, RGB, `[0,1]` scale (no mean/std normalization)
- Output: `1x1x224x224` float32, **relative** depth (not metric — see Caveats)
- Trained on NYU Depth V2 (indoor scenes)

## Directory layout

```
tools/model_conversion/fastdepth/
├── pyproject.toml            # uv project (onnx, onnxsim, numpy, Pillow)
├── scripts/
│   ├── export_fastdepth.py       # raw ONNX -> onnxsim -> opset17 -> fastdepth_224.onnx
│   ├── check_onnx_version.py     # IR/opset compatibility check (+ optional downgrade)
│   ├── download_coco_calib.py    # calibration image set (see "校准数据" below)
│   └── convert_to_cvimodel.sh    # run inside sophgo/tpuc_dev:v3.1: model_transform -> run_calibration -> model_deploy
├── pretrained/                # (gitignored) downloaded raw ONNX, not committed
├── calib_set/                 # (gitignored) calibration images, not committed
├── fastdepth_224.onnx         # final onnxsim'd, opset-17 ONNX (deliverable)
└── fastdepth_224_int8.cvimodel # final INT8 cvimodel for cv181x (deliverable)
```

## 模型来源 (weight source)

任务参考的官方权重（`dwofk/fast-depth` 的
`mobilenet-nnconv5dw-skipadd-pruned.pth.tar`）托管在
`http://datasets.lids.mit.edu/fastdepth/results/`。**该服务器从本机直连和走本机
HTTP 代理（Clash Verge, 127.0.0.1:7897）均连接超时（15s 无响应），判定为不可达**
（不是 Google Drive，是 MIT LIDS 自建服务器，2019 年论文附带资源，大概率已下线或对
本网络不可达）。

按任务指示的 fallback 顺序，改用 HuggingFace 搜索定位到
[`STMicroelectronics/fastdepth`](https://huggingface.co/STMicroelectronics/fastdepth)，
其 README 指出模型转换自 **PINTO0309/PINTO_model_zoo 的 `146_FastDepth`**
条目（同一个 `dwofk/fast-depth` checkpoint，经 `openvino2tensorflow` 导出）。
该条目的 `download.sh` 指向：

```
https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/146_FastDepth/resources.tar.gz  (724MB, 直连可达)
  -> saved_model_224x224/fast_depth_224x224.onnx   (已是 1x3x224x224 固定输入, opset 11, IR 6)
```

这个 ONNX 图本身已经只有 `Conv/Clip/Relu/Resize/Add`（含 `Constant`），与参考规格
的算子要求完全一致，无需从 PyTorch 重新导出。`scripts/export_fastdepth.py` 在此
基础上做：onnxsim 一遍 → opset 11→17 升级 → onnxsim 二遍 → 校验最终算子集合。

复现步骤：
```bash
mkdir -p pretrained
curl -sL -o pretrained/resources.tar.gz \
  https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/146_FastDepth/resources.tar.gz
tar -xzf pretrained/resources.tar.gz saved_model_224x224/fast_depth_224x224.onnx
mv saved_model_224x224/fast_depth_224x224.onnx pretrained/fast_depth_224x224_raw.onnx
rm -rf saved_model_224x224 pretrained/resources.tar.gz   # 只留需要的文件，省 700MB+

uv sync
uv run python scripts/export_fastdepth.py \
  --input pretrained/fast_depth_224x224_raw.onnx \
  --output fastdepth_224.onnx
```

## BF16：不需要校准（推荐路径）

**BF16 不需要校准表**，因此不需要校准集，也就不存在校准集选错的风险。BF16 逐张量
线性映射，指数位与 FP32 同宽，动态范围直接覆盖；INT8 只有 256 个格子，必须先统计
各层激活的实际取值范围才知道格子怎么摆。

直接复用 `model_transform` 产出的 `.mlir`，跳过 `run_calibration`：

```bash
docker run --rm --network host -v "$PWD":/workspace sophgo/tpuc_dev:v3.1 bash -c "
cd /workspace/model_workspace && pip install -q tpu_mlir[all]==1.7 &&
model_deploy.py --mlir fastdepth_224.mlir --quantize BF16 --processor cv181x \
  --test_input fastdepth_224_in_f32.npz \
  --test_reference fastdepth_224_top_outputs.npz \
  --model fastdepth_224_bf16.cvimodel"
```

约 40 秒完成（INT8 那条链路约 3 分钟，绝大部分耗在校准）。

### BF16 vs INT8 实测对照

| | BF16（默认） | INT8 |
|---|---|---|
| 校准集 | **不需要** | 200 张 COCO val2017 |
| `.cvimodel` | 2.9 MB | 1.4 MB |
| Need ION | 6.69 MB | 3.91 MB |
| cosine（vs float32） | 0.999998 | 0.999502 |
| **SQNR（vs float32）** | **39.98 dB** | **16.38 dB** |
| 推理（wiki 参考） | 约 19 ms | 约 18 ms |
| 转换耗时 | 约 40 秒 | 约 3 分钟 |

**该看的是 SQNR，不是 cosine。** 两者 cosine 都是 0.999x，但 cosine 对整体形状敏感、
对幅值误差不敏感；SQNR 直接衡量量化噪声功率，两者差 23.6 dB。深度模型输出是连续
标量场，量化噪声会直接表现为深度抖动和局部远近翻转——同样的噪声在检测模型里会被
argmax 吃掉，在这里不会。

INT8 换来的 1.5 MB 体积和约 6% 速度，在 66.7 ms 帧周期和约 60 MB ION 预算下都不
构成约束。**因此默认发 BF16。** 只有在确实需要极致体积/速度、且愿意用目标设备在
目标场景实拍的图重新校准时，才走 INT8。

## 校准数据 (calibration set) —— 仅 INT8 需要

> 走 BF16 可以整节跳过。

参考规格建议用 NYU Depth V2 / DIODE 的室内场景图。**订正**：初次转换时判定这两个源不可达，事后实测是错的——
NYU `nyu_depth_v2_labeled.mat`（2.8 GB）实测 1.65 MB/s，
DIODE `val.tar.gz`（2.58 GiB）实测 1.80 MB/s，均无需注册、支持断点续传：

```
https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
https://diode-dataset.s3.amazonaws.com/val.tar.gz
```

真正不可达的只有 FastDepth 预处理版所在的 `datasets.lids.mit.edu`（返回 000）。
当时因这个错误判断改用 `scripts/download_coco_calib.py`（照抄
`recamera_yolo_detection/scripts/download_coco_calib.py` 的模式）从
`images.cocodataset.org` 下载 COCO val2017 的 500 张随机通用场景图。

**已知取舍**：COCO 图片不是室内场景，和 FastDepth 训练分布（NYU indoor）不完全
匹配。INT8 校准只统计激活值的分布范围（min/max/直方图），不需要深度标签，用
通用自然图片校准在工程上可行，但理论上不如域内（室内）图片精确。如果后续要
优化精度，应换成真实室内/reCamera 拍摄的图片重新跑校准。

```bash
uv run python scripts/download_coco_calib.py --count 500 --output-dir calib_set
```

## 转换 (ONNX -> cvimodel)

在 `sophgo/tpuc_dev:v3.1` 容器内运行（工作目录挂载为 `/workspace`）：

```bash
docker run --rm --name fastdepth_convert \
  -v /Users/harvest/project/recamera/model_conversion/recamera_fastdepth:/workspace \
  sophgo/tpuc_dev:v3.1 \
  bash -c "cd /workspace && pip install -q tpu_mlir[all]==1.7 && \
           bash scripts/convert_to_cvimodel.sh --input-num 200"
```

`convert_to_cvimodel.sh` 依次执行：
1. `model_transform` — ONNX -> MLIR，`--mean 0,0,0 --scale 1/255,1/255,1/255 --pixel_format rgb`
   （**不带 `--keep_aspect_ratio`**，即默认 `keep_aspect_ratio=False` = 拉伸 resize
   到 224x224，见下方"已知坑 #1"）
2. `run_calibration` — INT8 校准表，`--input_num 200`（实测值，见下）
3. `model_deploy` — MLIR -> INT8 cvimodel，`--tolerance 0.85,0.45`
4. `model_tool --info` — 打印产物信息（Need ION 等）

本任务执行时发现容器 `yolo_convert` 已不存在（`docker ps -a` 无此容器，
`sophgo/tpuc_dev:v3.1` 镜像也未缓存本机），已重新 `docker pull` 并新建
`fastdepth_convert` 容器完成转换。

**实际用的 input_num：200，不是 500。** 先按参考规格的 500 张跑（容器名
`fastdepth_convert`），`run_calibration` 在 op 7/47（`258_Clip`）附近无任何报错
/ Traceback 静默停止输出，`docker run --rm` 最终以 exit code 1 结束，且
`model_workspace/` 下未生成 `fastdepth_224_calib_table` —— 现象与全局 CLAUDE.md
记录的"Mac Docker 校准静默 OOM"一致（这次是在 amd64 镜像跑在 arm64 Mac 上，
QEMU 模拟层内存开销更大，比 op 130-150 更早触发）。按已知 mitigation 把
`--input-num` 降到 200 后（容器名 `fastdepth_convert2`），校准在约 1 分钟内跑完，
`run_calibration` 进程峰值 RSS 约 2GB（`ps aux` 实测），随后 `model_deploy` 顺利
产出 `fastdepth_224_int8.cvimodel`（exit code 0）。原始日志见
`convert_log_500.txt`（失败，截断在 op 7/47）和 `convert_log_200.txt`（成功，
完整跑到 `model_tool --info`）。

## 已知坑 / caveats

1. **calib 预处理是拉伸 (stretch)，不是 letterbox，且和设备端一致**：
   `convert_log_200.txt` 里 `model_transform` 打印的 preprocess 配置是
   `keep_aspect_ratio : False`（`keep_ratio_mode: letterbox` 这一行是未生效的
   默认值，因为 `keep_aspect_ratio=False` 时不启用）。也就是说标定和推理时图片
   直接被拉伸到 224x224，不保长宽比、不补灰边。**这一点必须和设备端应用的预处理
   保持一致**——reCamera 推理通道如果做的是"去灰边后拉伸到 224x224"，两边一致，
   没有静默的预处理失配；如果设备端改成 letterbox/keep_aspect_ratio，必须同步改
   这里的 `model_transform` 参数重新转换，否则会静默掉精度。
2. **input_num=500 会被 Mac Docker 静默 OOM 杀掉，200 稳定**：见上文"转换"一节
   的实测记录。这次失败发生得比全局 CLAUDE.md 记录的"op 130-150"更早（op 7/47），
   推测和这次是 amd64 镜像在 arm64 Mac 上跑 QEMU 模拟（`docker run` 有
   `platform mismatch` 警告）有关，模拟层本身也吃内存。
3. **相对深度，非米制**：FastDepth 在 NYU Depth V2 上训练，输出是相对深度顺序，
   不是标定过的米制距离；室外场景的深度范围会被压缩失真。
4. **预处理已烘焙进 mlir**：`--mean 0,0,0 --scale 1/255` 表示"仅归一化到 [0,1]，
   无减均值/除标准差"，这组参数会被固化进 `.mlir`，设备端推理时无需再手动做
   normalize，只需 resize（拉伸，见坑#1）+ RGB。
5. **两道 gate 都要过**：Gate1 (量化 MLIR vs float32 参考，`model_deploy` 内部用
   `--tolerance 0.85,0.45` 跑 `npz_tool.py compare`) 实测
   `cosine_similarity=0.999502`，`min_similiarity=(0.9995, 0.9683, 16.38)`，
   PASSED。Gate2 (编译后 cvimodel vs 量化 tpu mlir，内部 tolerance 0.99,0.90)
   实测两个输出张量 `EQUAL [PASSED]`，`min_similiarity=(1.0, 1.0, inf)`，PASSED。
   两道 gate 原始输出见下方 EVIDENCE。

## 部署到 reCamera

仓库里已经带了转好的产物，直接推即可：

```bash
cd solutions/depth-estimation/model

# 默认版（BF16）——应用的默认模型路径指向它
scp fastdepth_224_bf16.cvimodel recamera@192.168.42.1:/tmp/
ssh recamera@192.168.42.1 "sudo mkdir -p /userdata/local/models && sudo mv /tmp/fastdepth_224_bf16.cvimodel /userdata/local/models/"

# INT8 版（可选，与 BF16 共存，用 -m 指定）
scp fastdepth_224_int8.cvimodel recamera@192.168.42.1:/tmp/
ssh recamera@192.168.42.1 "sudo mv /tmp/fastdepth_224_int8.cvimodel /userdata/local/models/"
```

## 产物

```
$ ls -lh *.onnx *.cvimodel
-rw-r--r--  5.2M  fastdepth_224.onnx
-rw-r--r--  2.9M  fastdepth_224_bf16.cvimodel   # 默认发布版
-rw-r--r--  1.4M  fastdepth_224_int8.cvimodel   # 进阶选项

$ grep "Need ION" convert_log_bf16.txt convert_log_200.txt
BF16: CviModel Need ION Memory Size: (6.69 MB)
INT8: CviModel Need ION Memory Size: (3.91 MB)
```

## EVIDENCE（2026-09-03 实测）

产物：

| 文件 | 大小 |
|---|---|
| `fastdepth_224.onnx` | 5.2 MB |
| `fastdepth_224_int8.cvimodel` | 1.4 MB |
| `fastdepth_224_calib_table` | 2.0 KB |

`CviModel Need ION Memory Size: (3.91 MB)`

ONNX 图（`fastdepth_224.onnx`，onnxsim 两遍 + opset 升级后）：

```
Counter({'Conv': 38, 'Clip': 27, 'Relu': 11, 'Resize': 5, 'Add': 3})
IR 6  opset 17
in   input.1  [1, 3, 224, 224]
out  424      [1, 1, 224, 224]
```

算子集合只有 `Conv/Clip/Relu/Resize/Add`，与参考规格一致，无 CV181x 不支持的算子。
IR 6 / opset 17 满足 IR<=8、opset<=17 的要求。

Gate1 — 量化 TPU MLIR vs float32 参考（`--tolerance 0.85,0.45`）：

```
[Success]: npz_tool.py compare fastdepth_224_cv181x_int8_sym_tpu_outputs.npz \
           fastdepth_224_top_outputs.npz --tolerance 0.85,0.45 --except - -vv
    cosine_similarity      = 0.999502
    euclidean_similarity   = 0.968287
    sqnr_similarity        = 16.378665
```

Gate2 — 编译后 cvimodel vs 量化 MLIR（`--tolerance 0.99,0.90`）：

```
[424_Relu_f32                    ]        EQUAL [PASSED]
[424_Relu                        ]        EQUAL [PASSED]
  2 equal, 0 close, 0 similar
  0 not equal, 0 not similar
npz compare PASSED.
Conversion Complete!
```

### 与 wiki 参考值的差异

| 指标 | wiki | 本次 |
|---|---|---|
| Gate1 cosine | 0.9997 | 0.999502 |
| cvimodel 大小 | 1.5 MB | 1.4 MB |
| 校准图片 | ~500 张（reCamera 实拍 + DIODE） | 200 张 COCO val2017 |

cosine 略低于 wiki，与校准集的差异方向一致（通用场景 vs 域内室内场景，且张数更少）。
数值层面差距很小，但**深度图的实际观感质量必须在真机上目视验证**——cosine 高
不等于深度结构对。若室内表现不佳，第一优先的改进是换成 reCamera 实拍的室内图
重新校准，而不是调量化参数。

### BF16 gate（无校准表，2026-09-03 实测）

```
[Success]: npz_tool.py compare fastdepth_224_cv181x_bf16_tpu_outputs.npz \
           fastdepth_224_top_outputs.npz --tolerance 0.8,0.5 --except - -vv
    cosine_similarity      = 0.999998
    euclidean_similarity   = 0.997957
    sqnr_similarity        = 39.979298
[424_Relu]      SIMILAR [PASSED]
  0 equal, 0 close, 2 similar / 0 not equal, 0 not similar
npz compare PASSED.

[Success]: npz_tool.py compare fastdepth_224_cv181x_bf16_model_outputs.npz \
           fastdepth_224_cv181x_bf16_tpu_outputs.npz --tolerance 0.99,0.90 --except - -vv

CviModel Need ION Memory Size: (6.69 MB)      # BF16
CviModel Need ION Memory Size: (3.91 MB)      # INT8 对照
```

全程未生成也未使用校准表。日志 `convert_log_bf16.txt`。

### 未做

- 真机延迟实测（wiki 报 INT8 18.38 ms / 54.4 FPS，本次未复现）
- BF16 与 INT8 在真机上的深度图观感对比
- 与 YOLO11n 共驻的 ION 与流水线 FPS 实测
