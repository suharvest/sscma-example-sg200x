#ifndef _ONVIF_META_H_
#define _ONVIF_META_H_

#include <cstdint>
#include <string>
#include <vector>

/*
 * onvif_meta: inference results in ONVIF's analytics data model.
 *
 * ONVIF 22.12 standardised a JSON representation of analytics metadata and an
 * MQTT topic layout for it (Analytics Service Spec 26.06 section 5.5). That is
 * almost exactly what the gallery applications already publish, so adopting it
 * costs a serialiser and buys "conforms to the ONVIF analytics metadata
 * representation" -- without implementing any ONVIF service, buying any
 * membership, or touching the video path.
 *
 * This header is the shared data model, deliberately separate from any
 * transport. Today it is serialised to JSON and published over MQTT; when the
 * RTSP metadata track lands (ONVIF Profile T section 7.13 makes it mandatory)
 * the same structures serialise to XML with no change at the call sites. Keep
 * it that way: no MQTT, no XML writer and no RTSP in this header.
 *
 * What it deliberately does NOT do: implement ONVIF services, claim
 * conformance, or let anyone print an ONVIF logo. Consuming clients that
 * matter here (Milestone, Genetec) parse the metadata itself and do not check
 * for a conformance mark.
 */

/* One class hypothesis. `type` is free text by design: tt:ClassDescriptor/Type
 * is a StringLikelihood, not an enum, and the schema explicitly allows vendor
 * values. Prefer the standard tt:ObjectType spellings where they fit --
 *   Animal, HumanFace, Human, Bicycle, Vehicle, LicensePlate, Bike, Barcode,
 *   Fire, Smoke
 * -- and invent names only for things ONVIF has no word for (weather classes,
 * OCR text). */
struct onvif_class_t {
    std::string type;
    float likelihood = 0.f;
};

/*
 * Face attributes. ONVIF models these first-class (tt:HumanFace, with the
 * fc: humanface namespace), so a face detector maps onto the standard rather
 * than needing vendor extensions -- unlike emotion or race, which ONVIF has no
 * word for and which belong in onvif_object_t::extensions.
 *
 * Gender uses ONVIF's spelling, "Male"/"Female", not the model's lowercase.
 * Age is a range because that is what fc:Age is (tt:IntRange); a point
 * estimate goes in as min == max.
 */
struct onvif_humanface_t {
    bool present = false;
    std::string gender;
    int age_min = -1;
    int age_max = -1;
};

/*
 * One detected object.
 *
 * Coordinates are CENTER-based pixels in the source frame with y pointing
 * DOWN, i.e. exactly what the models and debug_stream_box_t already produce.
 * The serialiser converts to ONVIF's convention; call sites must not
 * pre-convert. See the note on the ONVIF coordinate system in onvif_meta.cpp
 * -- getting this wrong renders every box upside down, and it looks plausible
 * enough to ship.
 */
struct onvif_object_t {
    int id = 0;
    float cx = 0.f, cy = 0.f, w = 0.f, h = 0.f;
    std::vector<onvif_class_t> classes;

    /* Parent object id, 0 for none. Used for containment, e.g. a licence plate
     * inside a vehicle (ONVIF tt:Object/@Parent). */
    int parent = 0;

    /* Barcode payload; when set, emits tt:BarcodeInfo. ONVIF has a first-class
     * element for this, so qrcode-reader maps cleanly. `barcode_type` takes
     * tt:BarcodeType spellings, e.g. "QRCode". */
    std::string barcode_data;
    std::string barcode_type;

    /* Face attributes; when present emits tt:HumanFace. */
    onvif_humanface_t face;

    /* Vendor extensions with no ONVIF equivalent (emotion, OCR text, ...).
     * Emitted under the recam: namespace. Keys must be XML-name safe. */
    std::vector<std::pair<std::string, std::string>> extensions;
};

/* One analysed frame. */
struct onvif_frame_t {
    uint64_t utc_ms = 0;   /* wall clock; see the note about 1970 below */
    std::string source;    /* analytics module name, e.g. "YoloDetector" */
    int frame_w = 0;       /* source frame size, needed for normalisation */
    int frame_h = 0;
    std::vector<onvif_object_t> objects;
};

/*
 * Serialise to the ONVIF JSON representation (Analytics Spec 26.06 5.5.3):
 *   {"Frame":[{"@UtcTime":..,"@Source":..,"Transformation":{..},"Object":[..]}]}
 *
 * An empty object list is valid and worth sending periodically: the spec asks
 * for a regular scene description even with nothing in it, so a client can
 * tell "analytics running, nothing seen" from "analytics dead".
 */
std::string onvif_meta_to_json(const onvif_frame_t& frame);

/*
 * MQTT topic per Analytics Spec 26.06 5.5.2:
 *   <topic_prefix>/onvif-mj/VideoAnalytics/<profile>/<module>
 *
 * This is an ADDITIONAL topic. The existing recamera/<app>/results contract is
 * consumed by SenseCraft (draw_weather.js among others) and must not change.
 */
std::string onvif_meta_topic(const std::string& topic_prefix,
                             const std::string& profile,
                             const std::string& module);

/* UTC timestamp in the format ONVIF uses, e.g. "2026-07-25T03:14:57.321Z".
 *
 * Note for whoever debugs this on hardware: reCamera has no RTC driver loaded
 * by default and boots at 1970, so this can legitimately emit 1970 timestamps.
 * That is a device configuration issue, not a bug here. */
std::string onvif_meta_utc(uint64_t unix_ms);

#endif /* _ONVIF_META_H_ */
