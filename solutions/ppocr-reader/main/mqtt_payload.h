#ifndef _PPOCR_MQTT_PAYLOAD_H_
#define _PPOCR_MQTT_PAYLOAD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "ocr_pipeline.h"

namespace ppocr {

// Build the results JSON payload published to the MQTT results topic.
// Text boxes are normalized against frame_width/frame_height.
std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                            const std::vector<OcrResult>& results,
                            const OcrTimings& timings,
                            int frame_width, int frame_height);

}  // namespace ppocr

#endif  // _PPOCR_MQTT_PAYLOAD_H_
