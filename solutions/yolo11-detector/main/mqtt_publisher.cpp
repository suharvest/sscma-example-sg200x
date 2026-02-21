#include "mqtt_publisher.h"

#include <sstream>
#include <iomanip>
#include <ctime>

#define TAG "MqttPublisher"

// Use SSCMA logging macros
#include <sscma.h>

namespace yolo11 {

MqttPublisher::MqttPublisher()
    : client_(nullptr),
      connected_(false),
      initialized_(false) {}

MqttPublisher::~MqttPublisher() {
    deinit();
}

bool MqttPublisher::init(const MqttConfig& config) {
    if (initialized_.load()) {
        MA_LOGW(TAG, "Already initialized");
        return true;
    }

    config_ = config;

    // Initialize mosquitto library
    mosquitto_lib_init();

    // Create client
    client_ = mosquitto_new(config_.client_id.c_str(), true, this);
    if (!client_) {
        MA_LOGE(TAG, "Failed to create MQTT client");
        return false;
    }

    // Set callbacks
    mosquitto_connect_callback_set(client_, onConnectCallback);
    mosquitto_disconnect_callback_set(client_, onDisconnectCallback);

    // Set credentials if provided
    if (!config_.username.empty() && !config_.password.empty()) {
        mosquitto_username_pw_set(client_, config_.username.c_str(),
                                   config_.password.c_str());
    }

    // Set reconnect delay
    mosquitto_reconnect_delay_set(client_, 2, 30, true);

    // Start network loop thread
    int rc = mosquitto_loop_start(client_);
    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGE(TAG, "Failed to start MQTT loop: %d", rc);
        mosquitto_destroy(client_);
        client_ = nullptr;
        return false;
    }

    // Connect to broker
    rc = mosquitto_connect(client_, config_.host.c_str(), config_.port, 60);
    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGW(TAG, "Initial connection failed (will retry): %d", rc);
        // Don't return false - mosquitto will auto-reconnect
    }

    initialized_.store(true);
    MA_LOGI(TAG, "MQTT publisher initialized: %s:%d", config_.host.c_str(), config_.port);
    MA_LOGI(TAG, "Publishing to topic: %s", config_.topic.c_str());

    return true;
}

void MqttPublisher::deinit() {
    if (!initialized_.load()) {
        return;
    }

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

    MA_LOGI(TAG, "MQTT publisher deinitialized");
}

void MqttPublisher::onConnectCallback(struct mosquitto* mosq, void* obj, int rc) {
    MqttPublisher* self = static_cast<MqttPublisher*>(obj);
    if (self) {
        self->onConnect(rc);
    }
}

void MqttPublisher::onDisconnectCallback(struct mosquitto* mosq, void* obj, int rc) {
    MqttPublisher* self = static_cast<MqttPublisher*>(obj);
    if (self) {
        self->onDisconnect(rc);
    }
}

void MqttPublisher::onConnect(int rc) {
    if (rc == 0) {
        connected_.store(true);
        MA_LOGI(TAG, "Connected to MQTT broker");
    } else {
        MA_LOGE(TAG, "Connection failed with code: %d", rc);
    }
}

void MqttPublisher::onDisconnect(int rc) {
    connected_.store(false);
    if (rc != 0) {
        MA_LOGW(TAG, "Unexpected disconnect: %d", rc);
    } else {
        MA_LOGI(TAG, "Disconnected from MQTT broker");
    }
}

bool MqttPublisher::publish(const std::string& topic, const std::string& payload) {
    if (!initialized_.load() || !client_) {
        return false;
    }

    if (!connected_.load()) {
        MA_LOGV(TAG, "Not connected, message will be queued");
    }

    int rc = mosquitto_publish(client_, nullptr, topic.c_str(),
                                static_cast<int>(payload.size()),
                                payload.data(), config_.qos, config_.retain);

    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGE(TAG, "Publish failed: %d", rc);
        return false;
    }

    return true;
}

std::string MqttPublisher::buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                                            const std::vector<Detection>& detections,
                                            float inference_time_ms) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(4);

    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":" << std::setprecision(1) << inference_time_ms << ",";
    json << "\"detection_count\":" << detections.size() << ",";
    // Use frame_width/height = 1 since bbox coordinates are normalized (0-1)
    json << "\"frame_width\":1,";
    json << "\"frame_height\":1,";
    json << "\"detections\":[";

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];

        if (i > 0) json << ",";

        json << "{";
        json << "\"id\":" << det.id << ",";
        json << "\"class_id\":" << det.class_id << ",";
        json << "\"class_name\":\"" << Yolo11Detector::getClassName(det.class_id) << "\",";
        json << "\"confidence\":" << std::setprecision(3) << det.confidence << ",";

        // Bounding box as array [x, y, w, h] with TOP-LEFT corner coordinates (normalized)
        // Convert from center coordinates: top_left_x = center_x - width/2
        float top_left_x = det.x - det.w / 2.0f;
        float top_left_y = det.y - det.h / 2.0f;
        json << "\"bbox\":[" << std::setprecision(4)
             << top_left_x << "," << top_left_y << ","
             << det.w << "," << det.h << "]";

        json << "}";
    }

    json << "]";
    json << "}";

    return json.str();
}

bool MqttPublisher::publishResults(uint64_t timestamp_ms, uint32_t frame_id,
                                    const std::vector<Detection>& detections,
                                    float inference_time_ms) {
    std::string payload = buildResultJson(timestamp_ms, frame_id, detections, inference_time_ms);
    return publish(config_.topic, payload);
}

std::string MqttPublisher::buildTrackingJson(uint64_t timestamp_ms, uint32_t frame_id,
                                              const std::vector<TrackedPerson>& persons,
                                              const StateCount& counts,
                                              float inference_time_ms) {
    std::ostringstream json;
    json << std::fixed;

    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":" << std::setprecision(1) << inference_time_ms << ",";
    // Use frame_width/height = 1 since bbox coordinates are normalized (0-1)
    json << "\"frame_width\":1,";
    json << "\"frame_height\":1,";

    // Zone occupancy summary
    json << "\"zone_occupancy\":{";
    json << "\"total\":" << counts.total << ",";
    json << "\"browsing\":" << counts.browsing << ",";
    json << "\"engaged\":" << counts.engaged << ",";
    json << "\"assistance\":" << counts.assistance;
    json << "},";

    // Person array
    json << "\"persons\":[";

    for (size_t i = 0; i < persons.size(); ++i) {
        const auto& person = persons[i];

        if (i > 0) json << ",";

        json << "{";
        json << "\"track_id\":" << person.track_id << ",";
        json << "\"confidence\":" << std::setprecision(3) << person.detection.confidence << ",";

        // Bounding box as array [x, y, w, h] with TOP-LEFT corner coordinates (normalized)
        float top_left_x = person.detection.x - person.detection.w / 2.0f;
        float top_left_y = person.detection.y - person.detection.h / 2.0f;
        json << "\"bbox\":[" << std::setprecision(4)
             << top_left_x << "," << top_left_y << ","
             << person.detection.w << "," << person.detection.h << "],";

        // Velocity info
        json << "\"speed_px_s\":" << std::setprecision(1) << person.speed_px_s << ",";
        json << "\"speed_normalized\":" << std::setprecision(1) << person.speed_normalized << ",";

        // Dwell state
        json << "\"state\":\"" << getDwellStateName(person.dwell_state) << "\",";
        json << "\"dwell_duration_sec\":" << std::setprecision(1) << person.dwell_duration_sec;

        json << "}";
    }

    json << "]";
    json << "}";

    return json.str();
}

bool MqttPublisher::publishTrackingResults(uint64_t timestamp_ms, uint32_t frame_id,
                                            const std::vector<TrackedPerson>& persons,
                                            const StateCount& counts,
                                            float inference_time_ms) {
    std::string payload = buildTrackingJson(timestamp_ms, frame_id, persons, counts, inference_time_ms);
    return publish(config_.topic, payload);
}

}  // namespace yolo11
