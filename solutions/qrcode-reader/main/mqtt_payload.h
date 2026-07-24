#ifndef _MQTT_PAYLOAD_H_
#define _MQTT_PAYLOAD_H_

#include <cstdint>
#include <string>
#include <vector>

namespace qrcode_reader {

// One decoded QR code. `points` are the four corners (top-left, clockwise)
// already remapped into the display frame and normalized to [0,1].
struct QrCode {
    std::string text;
    float points[4][2];
};

// Build the results JSON published to the MQTT results topic.
//
// EXTERNAL CONTRACT:
//   {"type":"qrcode","frame":N,"qr_found":bool,"detect_cost_ms":F,
//    "codes":[{"text":"..","points":[[x,y],[x,y],[x,y],[x,y]]},..]}
// `text` is JSON-escaped; `points` are normalized (4 decimals).
std::string buildResultJson(uint64_t frame_id,
                            bool qr_found,
                            double detect_cost_ms,
                            const std::vector<QrCode>& codes);

// Build the `"qrcodes":[...]` fragment appended verbatim as an extra top-level
// member of the debug /results envelope (same escaping / normalization).
// Returned WITHOUT a trailing comma and without surrounding braces.
std::string buildQrcodesExtra(const std::vector<QrCode>& codes);

}  // namespace qrcode_reader

#endif  // _MQTT_PAYLOAD_H_
