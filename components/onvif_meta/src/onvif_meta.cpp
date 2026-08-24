#include "onvif_meta.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <sstream>

/*
 * ONVIF coordinate system (Analytics Service Spec 26.06, section 5.2.2,
 * Figure 2) -- the single most error-prone part of this file:
 *
 *   - normalised range is -1..+1, NOT 0..1
 *   - the origin is the CENTRE of the frame
 *   - x runs left(-1) to right(+1), same as pixels
 *   - y runs bottom(-1) to top(+1), i.e. UP, the OPPOSITE of image pixels
 *
 * Because y is inverted, a valid tt:Rectangle always has top > bottom.
 *
 * Rather than normalise here, we emit pixel coordinates plus a per-frame
 * Transformation, which is what ONVIF's own examples do and what keeps the
 * numbers readable when someone tcpdumps this:
 *
 *     q = p * scale + translate      with translate = (-1,-1), scale = (2/W, 2/H)
 *
 * That maps px=0 -> -1 and px=W -> +1 correctly for x. For y it would map
 * py=0 (image TOP) to -1 (ONVIF BOTTOM), flipping the picture, so y is
 * mirrored into the transform's input space first:
 *
 *     y' = H - y
 *
 * With that, a box centred at (cx, cy) with size (w, h) in y-down pixels:
 *
 *     left   = cx - w/2          right  = cx + w/2
 *     top    = (H - cy) + h/2    bottom = (H - cy) - h/2      (top > bottom)
 *     CoG    = (cx, H - cy)
 *
 * If boxes ever come out vertically mirrored on a VMS, this is where to look,
 * not at the model.
 */

namespace {

/* JSON string escaping. Detection labels and especially OCR text and barcode
 * payloads are arbitrary bytes from the outside world, so this is not
 * optional. */
std::string esc(const std::string& in)
{
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b";  break;
        case '\f': out += "\\f";  break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:
            if (c < 0x20) {
                char buf[7];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

/* Escape XML character data and attribute values. XML 1.0 does not permit
 * most ASCII control characters even as character references, so replace
 * those bytes rather than returning a document that no conforming parser can
 * consume. Input text is otherwise expected to be UTF-8, like std::string
 * values throughout the applications. */
std::string xml_esc(const std::string& in)
{
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
        case '&':  out += "&amp;";  break;
        case '<':  out += "&lt;";   break;
        case '>':  out += "&gt;";   break;
        case '"': out += "&quot;"; break;
        case '\'': out += "&apos;"; break;
        default:
            if (c < 0x20 && c != '\t' && c != '\n' && c != '\r') {
                out += "&#xFFFD;";
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

bool xml_local_name_safe(const std::string& name)
{
    if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name[0])) ||
        name[0] == '_')) {
        return false;
    }
    for (unsigned char c : name) {
        if (!(std::isalnum(c) || c == '_' || c == '-' || c == '.')) {
            return false;
        }
    }
    return true;
}

std::string num(float v, int prec = 2)
{
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v;
    return o.str();
}

} // namespace

std::string onvif_meta_utc(uint64_t unix_ms)
{
    const std::time_t secs = static_cast<std::time_t>(unix_ms / 1000ull);
    const unsigned ms = static_cast<unsigned>(unix_ms % 1000ull);
    std::tm tm {};
    gmtime_r(&secs, &tm);
    char buf[40];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03uZ",
        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
        tm.tm_hour, tm.tm_min, tm.tm_sec, ms);
    return buf;
}

std::string onvif_meta_topic(const std::string& topic_prefix,
    const std::string& profile, const std::string& module)
{
    return topic_prefix + "/onvif-mj/VideoAnalytics/" + profile + "/" + module;
}

std::string onvif_meta_to_json(const onvif_frame_t& frame)
{
    std::ostringstream j;
    j << "{\"Frame\":[{";
    j << "\"@UtcTime\":\"" << onvif_meta_utc(frame.utc_ms) << "\"";
    if (!frame.source.empty()) {
        j << ",\"@Source\":\"" << esc(frame.source) << "\"";
    }
    j << ",\"@context\":{\"recam\":\"http://www.seeedstudio.com/recamera/schema\"}";

    const bool has_dims = frame.frame_w > 0 && frame.frame_h > 0;
    if (has_dims) {
        // Pixel coordinates below are interpreted through this; see the note
        // at the top of the file for why y is mirrored rather than scaled by a
        // negative factor (a negative scale would also flip the meaning of
        // top/bottom and break the top > bottom invariant).
        j << ",\"Transformation\":{"
          << "\"Translate\":{\"@x\":-1.0,\"@y\":-1.0},"
          << "\"Scale\":{\"@x\":" << num(2.0f / frame.frame_w, 8)
          << ",\"@y\":" << num(2.0f / frame.frame_h, 8) << "}}";
    }

    j << ",\"Object\":[";
    for (size_t i = 0; i < frame.objects.size(); ++i) {
        const onvif_object_t& o = frame.objects[i];
        if (i) j << ",";
        j << "{\"@ObjectId\":" << o.id;
        if (o.parent != 0) {
            j << ",\"@Parent\":" << o.parent;
        }
        j << ",\"Appearance\":{";

        // Element order inside Appearance follows the XSD sequence, which JSON
        // does not enforce but the XML serialiser will have to; keeping the
        // two identical avoids a divergence later:
        //   Shape -> Class -> LicensePlateInfo -> HumanFace -> BarcodeInfo
        const float cog_y = has_dims ? (frame.frame_h - o.cy) : o.cy;
        j << "\"Shape\":{"
          << "\"BoundingBox\":{"
          << "\"@left\":" << num(o.cx - o.w / 2.f)
          << ",\"@top\":" << num(cog_y + o.h / 2.f)      // y is up: top > bottom
          << ",\"@right\":" << num(o.cx + o.w / 2.f)
          << ",\"@bottom\":" << num(cog_y - o.h / 2.f)
          << "},"
          // CenterOfGravity is mandatory in the XSD, not an optimisation.
          << "\"CenterOfGravity\":{\"@x\":" << num(o.cx)
          << ",\"@y\":" << num(cog_y) << "}}";

        if (!o.classes.empty()) {
            j << ",\"Class\":{\"Type\":[";
            for (size_t k = 0; k < o.classes.size(); ++k) {
                if (k) j << ",";
                j << "{\"@Likelihood\":" << num(o.classes[k].likelihood, 3)
                  << ",\"#text\":\"" << esc(o.classes[k].type) << "\"}";
            }
            j << "]}";
        }

        if (o.face.present) {
            j << ",\"HumanFace\":{";
            bool first = true;
            if (!o.face.gender.empty()) {
                j << "\"Gender\":\"" << esc(o.face.gender) << "\"";
                first = false;
            }
            if (o.face.age_min >= 0 && o.face.age_max >= 0) {
                if (!first) j << ",";
                j << "\"Age\":{\"Min\":" << o.face.age_min
                  << ",\"Max\":" << o.face.age_max << "}";
            }
            j << "}";
        }

        if (!o.barcode_data.empty()) {
            j << ",\"BarcodeInfo\":{\"Data\":\"" << esc(o.barcode_data) << "\"";
            if (!o.barcode_type.empty()) {
                j << ",\"Type\":\"" << esc(o.barcode_type) << "\"";
            }
            j << "}";
        }

        for (const auto& kv : o.extensions) {
            j << ",\"recam:" << esc(kv.first) << "\":\"" << esc(kv.second) << "\"";
        }

        j << "}}";
    }
    j << "]}]}";
    return j.str();
}

std::string onvif_meta_to_xml(const onvif_frame_t& frame)
{
    std::ostringstream x;
    x << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
      << "<tt:MetadataStream"
      << " xmlns:tt=\"http://www.onvif.org/ver10/schema\""
      << " xmlns:fc=\"http://www.onvif.org/ver20/analytics/humanface\""
      << " xmlns:recam=\"http://www.seeedstudio.com/recamera/schema\">"
      << "<tt:VideoAnalytics>"
      << "<tt:Frame UtcTime=\"" << onvif_meta_utc(frame.utc_ms) << "\"";
    if (!frame.source.empty()) {
        x << " Source=\"" << xml_esc(frame.source) << "\"";
    }
    x << ">";

    const bool has_dims = frame.frame_w > 0 && frame.frame_h > 0;
    if (has_dims) {
        x << "<tt:Transformation>"
          << "<tt:Translate x=\"-1.0\" y=\"-1.0\"/>"
          << "<tt:Scale x=\"" << num(2.0f / frame.frame_w, 8)
          << "\" y=\"" << num(2.0f / frame.frame_h, 8) << "\"/>"
          << "</tt:Transformation>";
    }

    for (const onvif_object_t& o : frame.objects) {
        x << "<tt:Object ObjectId=\"" << o.id << "\"";
        if (o.parent != 0) {
            x << " Parent=\"" << o.parent << "\"";
        }
        x << "><tt:Appearance>";

        const float cog_y = has_dims ? (frame.frame_h - o.cy) : o.cy;
        x << "<tt:Shape>"
          << "<tt:BoundingBox left=\"" << num(o.cx - o.w / 2.f)
          << "\" top=\"" << num(cog_y + o.h / 2.f)
          << "\" right=\"" << num(o.cx + o.w / 2.f)
          << "\" bottom=\"" << num(cog_y - o.h / 2.f) << "\"/>"
          << "<tt:CenterOfGravity x=\"" << num(o.cx)
          << "\" y=\"" << num(cog_y) << "\"/>"
          << "</tt:Shape>";

        if (!o.classes.empty()) {
            x << "<tt:Class>";
            for (const onvif_class_t& c : o.classes) {
                x << "<tt:Type Likelihood=\"" << num(c.likelihood, 3) << "\">"
                  << xml_esc(c.type) << "</tt:Type>";
            }
            x << "</tt:Class>";
        }

        if (o.face.present) {
            x << "<tt:HumanFace>";
            // fc:HumanFace is an XSD sequence: Age must precede Gender.
            if (o.face.age_min >= 0 && o.face.age_max >= 0) {
                x << "<fc:Age><tt:Min>" << o.face.age_min
                  << "</tt:Min><tt:Max>" << o.face.age_max
                  << "</tt:Max></fc:Age>";
            }
            if (!o.face.gender.empty()) {
                x << "<fc:Gender>" << xml_esc(o.face.gender)
                  << "</fc:Gender>";
            }
            x << "</tt:HumanFace>";
        }

        if (!o.barcode_data.empty()) {
            x << "<tt:BarcodeInfo><tt:Data>" << xml_esc(o.barcode_data)
              << "</tt:Data>";
            if (!o.barcode_type.empty()) {
                x << "<tt:Type>" << xml_esc(o.barcode_type) << "</tt:Type>";
            }
            x << "</tt:BarcodeInfo>";
        }

        // Appearance's trailing xs:any is the schema-defined vendor extension
        // point. Keys are required by the data model contract to be safe XML
        // local names; values still need ordinary XML character escaping.
        for (const auto& kv : o.extensions) {
            if (!xml_local_name_safe(kv.first)) continue;
            x << "<recam:" << kv.first << ">" << xml_esc(kv.second)
              << "</recam:" << kv.first << ">";
        }

        x << "</tt:Appearance></tt:Object>";
    }

    x << "</tt:Frame></tt:VideoAnalytics></tt:MetadataStream>";
    return x.str();
}
