#ifndef _MICRODUCK_FACE_LED_INSPECTOR_H_
#define _MICRODUCK_FACE_LED_INSPECTOR_H_

#include <cstdint>
#include <vector>

#include "app_config.h"
#include "result_payload.h"

namespace microduck {

// Inspects an RGB888 image buffer for Face LED optical metrics:
// - lit_pixels: number of lit LED spots / blobs detected in the ROI (expecting 12)
// - color_temperature_k / mean_rgb: average RGB and estimated CCT in lit spots
// - dark_leakage_lux: max residual intensity outside lit blobs or when all LEDs are off
// - uniformity_ratio: min/max intensity across lit blobs
// - determines overall_passed = (lit_pixels >= 10 && dark_leakage_lux < 25.0)
FaceLedFrameResult inspectFaceLed(const uint8_t* rgb_data, int width, int height, const AppConfig& cfg);

// Single-LED check, one frame (Python twin: tools/sim_qa/face_led.py, keep identical).
// cfg.face_led_layout is relative to the lens ring. Steps: find the ring (round hole in a
// warm-hue mask) -> translation + scale; refine scale (+-15 %), rotation (+-15 deg) and
// translation (+-0.3 r) by the spot response at the seven places; least-squares similarity
// (Umeyama) onto the local spot peaks plus the ring centre; residual > 0.35 r or no ring:
// not registered. Colour per place: pixel minus local background (16 samples at 0.75 r,
// warm samples left out).
FaceLedSlotFrame inspectFaceLedSlots(const uint8_t* rgb_data, int width, int height, const AppConfig& cfg);

// Lens ring (cx, cy, r_mid) in pixels; false when none.
bool findLensRing(const uint8_t* rgb_data, int width, int height, float& cx, float& cy, float& r);

// Helper to calculate Correlated Color Temperature (CCT) in Kelvin from RGB values
float calculateColorTemperature(float r, float g, float b);

}  // namespace microduck

#endif  // _MICRODUCK_FACE_LED_INSPECTOR_H_
