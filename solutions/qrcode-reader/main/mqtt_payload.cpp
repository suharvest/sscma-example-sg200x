#include "mqtt_payload.h"

#include <cstdio>
#include <iomanip>
#include <sstream>

namespace qrcode_reader {

// JSON-escape a string (payload text is inserted verbatim into the document,
// so escaping is the caller's responsibility).
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// Render `[{"text":"..","points":[[x,y]x4]},..]`.
static std::string render_codes(const std::vector<QrCode>& codes) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(4);
    json << "[";
    for (size_t i = 0; i < codes.size(); ++i) {
        const QrCode& c = codes[i];
        if (i > 0) json << ",";
        json << "{\"text\":\"" << json_escape(c.text) << "\",\"points\":[";
        for (int p = 0; p < 4; ++p) {
            if (p > 0) json << ",";
            json << "[" << c.points[p][0] << "," << c.points[p][1] << "]";
        }
        json << "]}";
    }
    json << "]";
    return json.str();
}

std::string buildResultJson(uint64_t frame_id,
                            bool qr_found,
                            double detect_cost_ms,
                            const std::vector<QrCode>& codes) {
    std::ostringstream json;
    json << "{";
    json << "\"type\":\"qrcode\",";
    json << "\"frame\":" << frame_id << ",";
    json << "\"qr_found\":" << (qr_found ? "true" : "false") << ",";
    json << "\"detect_cost_ms\":" << std::fixed << std::setprecision(2) << detect_cost_ms << ",";
    json << "\"codes\":" << render_codes(codes);
    json << "}";
    return json.str();
}

std::string buildQrcodesExtra(const std::vector<QrCode>& codes) {
    std::string out = "\"qrcodes\":";
    out += render_codes(codes);
    return out;
}

}  // namespace qrcode_reader
