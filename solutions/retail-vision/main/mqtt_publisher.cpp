#include "mqtt_publisher.h"

#include <sstream>
#include <iomanip>

#define TAG "MqttPublisher"

#include <sscma.h>

namespace retail_vision {

MqttPublisher::MqttPublisher()
    : client_(nullptr),
      connected_(false),
      initialized_(false) {}

MqttPublisher::~MqttPublisher() {
    deinit();
}

bool MqttPublisher::init(const MqttConfig& config) {
    if (initialized_.load()) {
        return true;
    }

    config_ = config;

    mosquitto_lib_init();

    client_ = mosquitto_new(config_.client_id.c_str(), true, this);
    if (!client_) {
        MA_LOGE(TAG, "Failed to create MQTT client");
        return false;
    }

    mosquitto_connect_callback_set(client_, onConnectCallback);
    mosquitto_disconnect_callback_set(client_, onDisconnectCallback);

    if (!config_.username.empty() && !config_.password.empty()) {
        mosquitto_username_pw_set(client_, config_.username.c_str(),
                                   config_.password.c_str());
    }

    mosquitto_reconnect_delay_set(client_, 2, 30, true);

    int rc = mosquitto_loop_start(client_);
    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGE(TAG, "Failed to start MQTT loop: %d", rc);
        mosquitto_destroy(client_);
        client_ = nullptr;
        return false;
    }

    rc = mosquitto_connect_async(client_, config_.host.c_str(), config_.port, 60);
    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGW(TAG, "Initial connection failed (will retry): %d", rc);
    }

    initialized_.store(true);
    MA_LOGI(TAG, "MQTT publisher initialized: %s:%d topic=%s",
            config_.host.c_str(), config_.port, config_.topic.c_str());

    return true;
}

void MqttPublisher::deinit() {
    if (!initialized_.load()) return;

    if (client_) {
        if (connected_.load()) {
            mosquitto_disconnect(client_);
        }
        mosquitto_loop_stop(client_, true);
        mosquitto_destroy(client_);
        client_ = nullptr;
    }

    mosquitto_lib_cleanup();
    initialized_.store(false);
    connected_.store(false);
}

void MqttPublisher::onConnectCallback(struct mosquitto* mosq, void* obj, int rc) {
    auto* self = static_cast<MqttPublisher*>(obj);
    if (self) self->onConnect(rc);
}

void MqttPublisher::onDisconnectCallback(struct mosquitto* mosq, void* obj, int rc) {
    auto* self = static_cast<MqttPublisher*>(obj);
    if (self) self->onDisconnect(rc);
}

void MqttPublisher::onConnect(int rc) {
    if (rc == 0) {
        connected_.store(true);
        MA_LOGI(TAG, "Connected to MQTT broker");
    } else {
        MA_LOGE(TAG, "Connection failed: %d", rc);
    }
}

void MqttPublisher::onDisconnect(int rc) {
    connected_.store(false);
    if (rc != 0) {
        MA_LOGW(TAG, "Unexpected disconnect: %d", rc);
    }
}

bool MqttPublisher::publish(const std::string& topic, const std::string& payload) {
    if (!initialized_.load() || !client_) return false;

    int rc = mosquitto_publish(client_, nullptr, topic.c_str(),
                                static_cast<int>(payload.size()),
                                payload.data(), config_.qos, config_.retain);
    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGE(TAG, "Publish failed: %d", rc);
        return false;
    }
    return true;
}

std::string MqttPublisher::buildVisionJson(uint64_t timestamp_ms, uint32_t frame_id,
                                            float fps, float inference_time_ms,
                                            const ZoneSnapshot& zone,
                                            const std::vector<TrackedPerson>& persons,
                                            int frame_width, int frame_height,
                                            int model_width, int model_height) {
    std::ostringstream j;
    j << std::fixed;

    float fw = static_cast<float>(frame_width);
    float fh = static_cast<float>(frame_height);
    float mw = static_cast<float>(model_width);
    float mh = static_cast<float>(model_height);

    // Letterbox correction: VPSS fits the source into model input preserving aspect ratio.
    // For 16:9 display → square model: width fills, height has padding.
    // Compute the content area within the model's input space.
    float display_aspect = fw / fh;
    float model_aspect = mw / mh;

    // Scale factor and padding in normalized [0,1] model space
    float scale_x = 1.0f, scale_y = 1.0f;
    float offset_x = 0.0f, offset_y = 0.0f;

    if (display_aspect > model_aspect) {
        // Landscape: width fills model, height is padded
        float content_h = mh * (model_aspect / display_aspect);  // e.g. 640*(1.0/1.778)=360
        scale_y = content_h / mh;       // 360/640 = 0.5625
        offset_y = (1.0f - scale_y) / 2.0f;  // (1-0.5625)/2 = 0.21875
    } else if (display_aspect < model_aspect) {
        // Portrait: height fills model, width is padded
        float content_w = mw * (display_aspect / model_aspect);
        scale_x = content_w / mw;
        offset_x = (1.0f - scale_x) / 2.0f;
    }

    j << "{";
    j << "\"timestamp\":" << timestamp_ms;
    j << ",\"frame_id\":" << frame_id;
    j << ",\"frame_width\":" << frame_width;
    j << ",\"frame_height\":" << frame_height;
    j << ",\"fps\":" << std::setprecision(1) << fps;
    j << ",\"inference_time_ms\":" << std::setprecision(1) << inference_time_ms;

    // Zone metrics
    j << ",\"zone\":{";
    j << "\"occupancy_count\":" << zone.occupancy_count;
    j << ",\"browsing_count\":" << zone.browsing_count;
    j << ",\"engaged_count\":" << zone.engaged_count;
    j << ",\"assist_count\":" << zone.assist_count;
    j << ",\"peak_customer\":" << zone.peak_customer;
    j << ",\"avg_dwell_time\":" << std::setprecision(1) << zone.avg_dwell_time;
    j << ",\"avg_engagement_time\":" << std::setprecision(1) << zone.avg_engagement_time;
    j << ",\"avg_velocity\":" << std::setprecision(2) << zone.avg_velocity;
    j << ",\"entry_count\":" << zone.entry_count;
    j << ",\"exit_count\":" << zone.exit_count;
    j << "}";

    // Persons array — display-normalized coords (top-left x,y + w,h in [0,1]), matching draw script format
    j << ",\"persons\":[";
    for (size_t i = 0; i < persons.size(); ++i) {
        const auto& p = persons[i];
        if (i > 0) j << ",";

        // Undo letterbox: convert from model-normalized [0,1] to display-normalized [0,1]
        float real_cx = (p.detection.x - offset_x) / scale_x;
        float real_cy = (p.detection.y - offset_y) / scale_y;
        float real_w  = p.detection.w / scale_x;
        float real_h  = p.detection.h / scale_y;

        // Top-left normalized coords
        float bx = real_cx - real_w / 2.0f;
        float by = real_cy - real_h / 2.0f;

        j << "{";
        j << "\"track_id\":" << p.track_id;
        j << ",\"confidence\":" << std::setprecision(2) << p.detection.score;

        j << ",\"bbox\":{";
        j << "\"x\":" << std::setprecision(4) << bx;
        j << ",\"y\":" << by;
        j << ",\"w\":" << real_w;
        j << ",\"h\":" << real_h;
        j << "}";

        j << ",\"velocity\":{";
        j << "\"vx\":" << std::setprecision(2) << (p.velocity_x / fw);
        j << ",\"vy\":" << std::setprecision(2) << (p.velocity_y / fh);
        j << ",\"speed_m_s\":" << std::setprecision(2) << p.speed_m_s;
        j << "}";

        j << ",\"state\":\"" << getDwellStateName(p.dwell_state) << "\"";
        j << ",\"dwell_duration\":" << std::setprecision(1) << p.dwell_duration_sec;

        j << "}";
    }
    j << "]";

    j << "}";

    return j.str();
}

bool MqttPublisher::publishVisionPayload(uint64_t timestamp_ms, uint32_t frame_id,
                                          float fps, float inference_time_ms,
                                          const ZoneSnapshot& zone,
                                          const std::vector<TrackedPerson>& persons,
                                          int frame_width, int frame_height,
                                          int model_width, int model_height) {
    std::string payload = buildVisionJson(timestamp_ms, frame_id, fps, inference_time_ms, zone, persons, frame_width, frame_height, model_width, model_height);
    return publish(config_.topic, payload);
}

}  // namespace retail_vision
