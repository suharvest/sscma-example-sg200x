#include "mqtt_publisher.h"

#include <sstream>
#include <iomanip>

#include <sscma.h>

#define TAG "MqttPublisher"

namespace detection_blur {

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

    rc = mosquitto_connect(client_, config_.host.c_str(), config_.port, 60);
    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGW(TAG, "Initial connection failed (will retry): %d", rc);
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
                                            const std::vector<DetectionBox>& detections,
                                            float inference_time_ms) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(3);

    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":" << inference_time_ms << ",";
    json << "\"count\":" << detections.size() << ",";
    json << "\"detections\":[";

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];

        if (i > 0) json << ",";

        json << "{";
        json << "\"target\":" << det.target << ",";
        json << "\"bbox\":{";
        json << "\"x\":" << det.x << ",";
        json << "\"y\":" << det.y << ",";
        json << "\"w\":" << det.w << ",";
        json << "\"h\":" << det.h;
        json << "},";
        json << "\"confidence\":" << det.score;
        json << "}";
    }

    json << "]";
    json << "}";

    return json.str();
}

bool MqttPublisher::publishResults(uint64_t timestamp_ms, uint32_t frame_id,
                                    const std::vector<DetectionBox>& detections,
                                    float inference_time_ms) {
    std::string payload = buildResultJson(timestamp_ms, frame_id, detections, inference_time_ms);
    return publish(config_.topic, payload);
}

}  // namespace detection_blur
