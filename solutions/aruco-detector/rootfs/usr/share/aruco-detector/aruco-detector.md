# ArUco Marker Detector for Robotic Arm (SG2002 RISC-V)

High-performance visual fiducial marker detector and 6-DoF pose estimator designed for robotic arms, visual servoing, and hand-eye calibration on Seeed reCamera (Sophgo SG2002 single-core 1.0 GHz RISC-V).

## Features
- **Zero-Copy Grayscale Pipeline**: Directly ingests hardware NV21 Y-plane from the VPSS image processing pipeline with zero CPU format conversion overhead.
- **Embedded AprilTag 3 Engine**: Lightweight, pure C99 implementation supporting standard ArUco dictionary families (`tagAruco4x4_50`, `tagAruco4x4_100`, `tagAruco4x4_250`, `tagAruco5x5_50`, `tagAruco6x6_250`) and AprilTag (`tag36h11`).
- **High FPS & Low Latency**: Runs at 25–30 FPS with decimated gradient clustering and sub-pixel edge refinement.
- **6-DoF Metric Pose Estimation**: Solves 3D translation ($tvec = [x, y, z]$ in meters) and orientation (Euler angles roll, pitch, yaw) directly in the camera coordinate frame.
- **MQTT & Web Console Stream**: Publishes structured JSON data over MQTT for ROS/ROS2 nodes and streams H.264 preview with corner highlights over WebSocket port 8001.

## MQTT Payload Format
Topic: `recamera/aruco-detector/results`
```json
{
  "type": "aruco_detection",
  "frame_id": 1024,
  "timestamp_ms": 1718000000123,
  "detect_cost_ms": 14.5,
  "marker_count": 1,
  "markers": [
    {
      "id": 12,
      "family": "aruco4x4_50",
      "center": [0.512, 0.485],
      "corners": [[0.45, 0.40], [0.58, 0.41], [0.57, 0.56], [0.44, 0.55]],
      "pose": {
        "tvec": [0.0150, -0.0420, 0.4500],
        "rvec_deg": [1.5, 3.2, 45.0],
        "error": 0.00012
      }
    }
  ]
}
```
