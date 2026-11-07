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

// Helper to calculate Correlated Color Temperature (CCT) in Kelvin from RGB values
float calculateColorTemperature(float r, float g, float b);

}  // namespace microduck

#endif  // _MICRODUCK_FACE_LED_INSPECTOR_H_
