# Robotic Arm Vision Companion Atomic Service (reCamera SG2002 RISC-V)

High-performance atomic vision service running on Seeed reCamera (SG2002 single-core 1.0 GHz RISC-V), designed for robotic arm visual servoing, hand-eye calibration, and precision grasping.

## Architectural Boundary Principles
**reCamera only provides low-level, high-performance atomic capabilities without hosting complex application orchestration or heavy Python runtimes:**
- High-level orchestration, VLM visual reasoning, and trajectory generation are handled remotely by host PC / cloud Agents.
- Once the pipeline logic is verified, it can be compiled into a hardened local binary running deterministically on the client.

## Core Atomic Capabilities & API Endpoints

| Protocol / Port | Endpoint | Mode | Description |
|---|---|---|---|
| **HTTP :8090** | `GET /api/inference` | Status | Returns `{"enabled":false,"fps":1,"active_fps":25,"idle_fps":1}` |
| **HTTP :8090** | `POST /api/inference` | Switch | Pass `{"enabled":true}` or `{"enabled":false}` to toggle standby rate |
| **HTTP :8090** | `POST /api/trigger` | **Atomic Trigger** | Single-shot capture & pose estimation; auto-restores to standby |
| **HTTP :8090** | `GET /api/results/latest` | Long poll | `?after=<id>&wait_ms=1000`, waits for fresh frame |
| **HTTP :80** | `GET /snapshot.jpg` | Snapshot | Raw JPEG image for remote VLM / multimodal Agents |
| **WS :8001** | `ws://{host}:8001/` | H.264 Stream | Low-latency 720P video preview stream (hardware decoded by JMuxer) |
| **WS :8001** | `ws://{host}:8001/results`| Telemetry | Continuous detection stream with pixel centers and 6-DoF poses |
| **MQTT :1883** | `recamera/aruco-detector/results` | Pub/Sub | MQTT topic broadcast for automation buses |

## Low Power & Thermal Efficiency
- **Standby Mode (Inference Disabled)**: Camera throttled to 1 FPS, pose estimation bypassed. SG2002 CPU load < 3%, case temperature steady at 38°C.
- **Active Mode (Triggered)**: Resumes to full 25 FPS with ~4ms processing latency per frame.
