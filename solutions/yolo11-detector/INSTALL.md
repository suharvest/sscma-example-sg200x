# YOLO11 Detector 部署指南

## 概述

YOLO11 Detector 是一个运行在 reCamera 上的实时目标检测与人员追踪应用，提供：
- YOLO11 目标检测（~100-120ms/帧，~8 FPS）
- DFL (Distribution Focal Loss) 后处理解码
- 人员追踪与停留状态分析
- RTSP 视频流输出
- MQTT 检测结果推送

## 系统要求

- **设备**: reCamera (cv181x SoC)
- **系统**: SysVinit (非 systemd)
- **网络**: 可通过 SSH 访问设备

## 文件准备

部署需要以下文件：

| 文件 | 说明 | 目标位置 |
|-----|------|---------|
| `yolo11-detector_0.1.0_riscv64.deb` | 应用程序包（含 init 脚本） | 安装到系统 |
| `yolo11n_detection_cv181x_int8.cvimodel` | YOLO11n 模型 (INT8) | `/userdata/local/models/` |

> **注意**: deb 包已包含 init 脚本，安装后自动部署到 `/etc/init.d/S92yolo11-detector`。
> 模型文件可从 supervisor 方案中获取：`solutions/supervisor/rootfs/usr/share/supervisor/models/yolo11n_detection_cv181x_int8.cvimodel`

## 一键部署脚本

将以下脚本保存为 `deploy.sh`，修改 `DEVICE_IP` 和 `DEVICE_PASS` 后执行：

```bash
#!/bin/bash
set -e

# ============ 配置区域 ============
DEVICE_IP="192.168.10.88"      # 设备 IP（有线或 USB: 192.168.42.1）
DEVICE_USER="recamera"
DEVICE_PASS="recamera.2"       # 默认密码，部分设备是 "recamera"

# 文件路径（相对于脚本目录）
DEB_FILE="yolo11-detector_0.1.0_riscv64.deb"
MODEL_FILE="yolo11n_detection_cv181x_int8.cvimodel"
# =================================

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "=== YOLO11 Detector 部署脚本 ==="
echo "目标设备: $DEVICE_IP"
echo ""

# 检查文件
for f in "$DEB_FILE" "$MODEL_FILE"; do
    if [ ! -f "$f" ]; then
        echo "错误: 找不到文件 $f"
        exit 1
    fi
done

# 1. 复制文件到设备
echo "[1/5] 复制文件到设备..."
sshpass -p "$DEVICE_PASS" scp $SSH_OPTS \
    "$DEB_FILE" "$MODEL_FILE" \
    ${DEVICE_USER}@${DEVICE_IP}:/tmp/

# 2. 安装和配置
echo "[2/5] 安装应用和配置..."
sshpass -p "$DEVICE_PASS" ssh $SSH_OPTS ${DEVICE_USER}@${DEVICE_IP} << 'REMOTE_SCRIPT'
# 停止冲突服务
echo "$DEVICE_PASS" | sudo -S /etc/init.d/S03node-red stop 2>/dev/null || true
echo "$DEVICE_PASS" | sudo -S /etc/init.d/S91sscma-node stop 2>/dev/null || true
echo "$DEVICE_PASS" | sudo -S /etc/init.d/S93sscma-supervisor stop 2>/dev/null || true
sleep 2

# 创建模型目录
echo "$DEVICE_PASS" | sudo -S mkdir -p /userdata/local/models

# 复制模型
echo "$DEVICE_PASS" | sudo -S cp /tmp/yolo11n_detection_cv181x_int8.cvimodel /userdata/local/models/

# 安装 deb 包（自动安装 init 脚本）
echo "$DEVICE_PASS" | sudo -S opkg install --force-reinstall /tmp/yolo11-detector_0.1.0_riscv64.deb
REMOTE_SCRIPT

# 3. 配置 Mosquitto 允许外部访问
echo "[3/5] 配置 MQTT..."
sshpass -p "$DEVICE_PASS" ssh $SSH_OPTS ${DEVICE_USER}@${DEVICE_IP} << 'REMOTE_SCRIPT'
# 检查是否已配置
if ! grep -q "listener 1883 0.0.0.0" /etc/mosquitto/mosquitto.conf 2>/dev/null; then
    echo "$DEVICE_PASS" | sudo -S sh -c 'echo "listener 1883 0.0.0.0" >> /etc/mosquitto/mosquitto.conf'
    echo "$DEVICE_PASS" | sudo -S sh -c 'echo "allow_anonymous true" >> /etc/mosquitto/mosquitto.conf'
    echo "$DEVICE_PASS" | sudo -S killall mosquitto 2>/dev/null || true
    sleep 1
    echo "$DEVICE_PASS" | sudo -S /usr/sbin/mosquitto -c /etc/mosquitto/mosquitto.conf -d
    echo "MQTT 已配置为允许外部访问"
else
    echo "MQTT 已配置"
fi
REMOTE_SCRIPT

# 4. 禁用冲突服务（可选但推荐）
echo "[4/5] 禁用冲突服务..."
sshpass -p "$DEVICE_PASS" ssh $SSH_OPTS ${DEVICE_USER}@${DEVICE_IP} << 'REMOTE_SCRIPT'
# 将 S 改为 K 禁用自启动
for svc in S03node-red S91sscma-node S93sscma-supervisor; do
    if [ -f "/etc/init.d/$svc" ]; then
        new_name=$(echo $svc | sed 's/^S/K/')
        echo "$DEVICE_PASS" | sudo -S mv "/etc/init.d/$svc" "/etc/init.d/$new_name" 2>/dev/null || true
        echo "已禁用: $svc -> $new_name"
    fi
done
REMOTE_SCRIPT

# 5. 启动服务
echo "[5/5] 启动 YOLO11 Detector..."
sshpass -p "$DEVICE_PASS" ssh $SSH_OPTS ${DEVICE_USER}@${DEVICE_IP} << 'REMOTE_SCRIPT'
echo "$DEVICE_PASS" | sudo -S /etc/init.d/S92yolo11-detector start
sleep 3
if ps aux | grep -v grep | grep yolo11-detector > /dev/null; then
    echo "服务启动成功!"
else
    echo "警告: 服务可能未启动，请检查日志"
fi
REMOTE_SCRIPT

echo ""
echo "=== 部署完成 ==="
echo ""
echo "访问方式:"
echo "  RTSP 流: rtsp://${DEVICE_IP}:8554/live0"
echo "  MQTT 主题: recamera/yolo11/detections"
echo ""
echo "管理命令 (SSH 登录后):"
echo "  启动: sudo /etc/init.d/S92yolo11-detector start"
echo "  停止: sudo /etc/init.d/S92yolo11-detector stop"
echo "  状态: sudo /etc/init.d/S92yolo11-detector status"
echo "  日志: tail -f /var/log/yolo11-detector.log"
```

## 手动部署步骤

如果不使用脚本，按以下步骤操作：

### 1. 复制文件

```bash
# 复制 deb 包和模型
scp yolo11-detector_0.1.0_riscv64.deb recamera@192.168.10.88:/tmp/
scp yolo11n_detection_cv181x_int8.cvimodel recamera@192.168.10.88:/tmp/
```

### 2. SSH 登录设备

```bash
ssh recamera@192.168.10.88
# 密码: recamera.2 或 recamera
```

### 3. 停止冲突服务

```bash
# 这些服务会占用摄像头，必须停止
sudo /etc/init.d/S03node-red stop
sudo /etc/init.d/S91sscma-node stop
sudo /etc/init.d/S93sscma-supervisor stop
```

### 4. 安装应用

```bash
# 创建模型目录
sudo mkdir -p /userdata/local/models

# 复制模型
sudo cp /tmp/yolo11n_detection_cv181x_int8.cvimodel /userdata/local/models/

# 安装 deb 包（自动安装 init 脚本到 /etc/init.d/）
sudo opkg install /tmp/yolo11-detector_0.1.0_riscv64.deb
```

### 5. 启动服务

```bash
sudo /etc/init.d/S92yolo11-detector start
```

## 自定义配置

创建 `/etc/yolo11-detector.conf` 可覆盖默认参数：

```bash
# YOLO11 Detector 配置文件
DAEMON_OPTS="--model /userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel \
    --conf-threshold 0.3 \
    --mqtt-topic recamera/yolo11/detections \
    --dwell-engaged 2.0 \
    --dwell-assist 30.0"
```

### 可用参数

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--model PATH` | 模型文件路径 | `/userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel` |
| `--conf-threshold` | 检测置信度阈值 | 0.25 |
| `--nms-threshold` | NMS 阈值 | 0.45 |
| `--mqtt-host` | MQTT 服务器地址 | localhost |
| `--mqtt-port` | MQTT 端口 | 1883 |
| `--mqtt-topic` | MQTT 主题 | recamera/yolo11/detections |
| `--dwell-speed` | 停留判定速度阈值 (px/s) | 10.0 |
| `--dwell-engaged` | ENGAGED 状态时间 (秒) | 1.5 |
| `--dwell-assist` | ASSISTANCE 状态时间 (秒) | 20.0 |
| `--no-rtsp` | 禁用 RTSP 流 | - |
| `--no-mqtt` | 禁用 MQTT | - |
| `--no-tracking` | 禁用人员追踪 | - |
| `-v, --verbose` | 详细日志 | - |

## 验证安装

### 检查服务状态

```bash
sudo /etc/init.d/S92yolo11-detector status
```

### 查看日志

```bash
tail -f /var/log/yolo11-detector.log
```

### 测试 RTSP 流

使用 VLC 或 ffplay 打开：
```
rtsp://192.168.10.88:8554/live0
```

### 测试 MQTT

```bash
# 在设备上
mosquitto_sub -h localhost -t "recamera/yolo11/detections" -C 1

# 或从外部（需已配置外部访问）
mosquitto_sub -h 192.168.10.88 -t "recamera/yolo11/detections" -C 1
```

### MQTT 消息格式

```json
{
    "timestamp": 1768969602957,
    "frame_id": 107,
    "inference_time_ms": 110.0,
    "zone_occupancy": {
        "total": 1,
        "browsing": 0,
        "engaged": 1,
        "assistance": 0
    },
    "persons": [{
        "track_id": 1,
        "confidence": 0.85,
        "bbox": {"x": 0.5, "y": 0.4, "w": 0.2, "h": 0.5},
        "speed_px_s": 0.2,
        "speed_normalized": 0.2,
        "state": "engaged",
        "dwell_duration_sec": 8.9
    }]
}
```

## YOLO26 vs YOLO11 对比

| 指标 | YOLO26 | YOLO11 |
|-----|--------|--------|
| BBox 通道数 | 4 (直接坐标) | 64 (4x16 DFL) |
| BBox 解码 | offset + exp | DFL softmax -> 距离 |
| 推理时间 | ~300ms/帧 | ~100-120ms/帧 |
| 帧率 | ~3.3 FPS | ~8 FPS |
| 模型大小 | ~3MB (INT8) | ~3MB (INT8) |

## 故障排除

### 问题: 服务启动失败

```bash
# 查看详细日志
cat /var/log/yolo11-detector.log

# 检查是否有进程占用摄像头
ps aux | grep -E 'sscma|node-red|face'

# 停止所有可能的冲突服务
sudo killall node-red sscma-node sscma-supervisor 2>/dev/null
```

### 问题: VPSS 错误 "Grp(0) is occupied"

```bash
# 重启设备清理 VPSS 状态
sudo reboot
```

### 问题: 权限错误 (TPU 初始化失败)

应用必须以 root 权限运行：

```bash
sudo /usr/local/bin/yolo11-detector --model /userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel
```

## 卸载

```bash
# 停止服务
sudo /etc/init.d/S92yolo11-detector stop

# 删除 init 脚本
sudo rm /etc/init.d/S92yolo11-detector

# 卸载 deb 包
sudo opkg remove yolo11-detector

# 删除模型（可选）
sudo rm /userdata/local/models/yolo11n_detection_cv181x_int8.cvimodel

# 恢复冲突服务（可选）
sudo mv /etc/init.d/K03node-red /etc/init.d/S03node-red
sudo mv /etc/init.d/K91sscma-node /etc/init.d/S91sscma-node
sudo mv /etc/init.d/K93sscma-supervisor /etc/init.d/S93sscma-supervisor
```

## 性能参考

| 指标 | 数值 |
|-----|------|
| 推理时间 | ~100-120ms/帧 |
| 帧率 | ~8 FPS |
| 模型大小 | ~3MB (INT8) |
| 输入分辨率 | 640x640 |
| RTSP 分辨率 | 1280x720 @ 15fps |
| DFL 解码 | 4 coords x 16 bins |
