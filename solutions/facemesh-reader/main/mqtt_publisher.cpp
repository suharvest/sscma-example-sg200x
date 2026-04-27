#include "mqtt_publisher.h"

#include <sstream>
#include <iomanip>
#include <ctime>

#include <sscma.h>

#include "facemesh_pipeline.h"  // for AnalyzedFace definition

#define TAG "MqttPublisher"

namespace facemesh_reader {

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
    MA_LOGI(TAG, "MQTT publisher deinitialized");
}

void MqttPublisher::onConnectCallback(struct mosquitto*, void* obj, int rc) {
    if (auto* self = static_cast<MqttPublisher*>(obj)) self->onConnect(rc);
}

void MqttPublisher::onDisconnectCallback(struct mosquitto*, void* obj, int rc) {
    if (auto* self = static_cast<MqttPublisher*>(obj)) self->onDisconnect(rc);
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
    if (!initialized_.load() || !client_) return false;

    int rc = mosquitto_publish(client_, nullptr, topic.c_str(),
                                static_cast<int>(payload.size()),
                                payload.data(), config_.qos, config_.retain);

    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGE(TAG, "Publish failed: rc=%d, connected=%d, topic=%s, size=%d",
                rc, connected_.load() ? 1 : 0, topic.c_str(), (int)payload.size());
        return false;
    }
    return true;
}

std::string MqttPublisher::buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                                            const std::vector<AnalyzedFace>& faces,
                                            float inference_time_ms,
                                            bool include_landmarks) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(4);

    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":" << inference_time_ms << ",";
    json << "\"face_count\":" << faces.size() << ",";
    json << "\"faces\":[";

    for (size_t i = 0; i < faces.size(); ++i) {
        const auto& af = faces[i];
        if (i > 0) json << ",";

        json << "{";
        json << "\"id\":" << af.face.id << ",";

        // Bounding box (normalized 0-1)
        json << "\"bbox\":{";
        json << "\"x\":" << af.face.x << ",";
        json << "\"y\":" << af.face.y << ",";
        json << "\"w\":" << af.face.w << ",";
        json << "\"h\":" << af.face.h;
        json << "},";
        json << "\"confidence\":" << af.face.score << ",";

        // EAR / MAR
        json << "\"left_ear\":" << af.metrics.left_ear << ",";
        json << "\"right_ear\":" << af.metrics.right_ear << ",";
        json << "\"ear\":" << af.metrics.avg_ear << ",";
        json << "\"mar\":" << af.metrics.mar << ",";
        json << "\"eyes_closed\":" << (af.metrics.eyes_closed ? "true" : "false") << ",";
        json << "\"mouth_open\":" << (af.metrics.mouth_open ? "true" : "false") << ",";
        json << "\"metrics_valid\":" << (af.metrics.valid ? "true" : "false");

        // ---- Phase 2: edge-autonomous drowsiness conclusion ----
        json << ",\"drowsiness\":{";
        json << "\"level\":" << af.drowsiness.drowsiness_level << ",";
        json << "\"state\":\"" << af.drowsiness.state << "\",";
        json << "\"perclos_pct\":" << af.drowsiness.perclos_pct << ",";
        json << "\"continuous_closure_sec\":" << af.drowsiness.continuous_closure_sec << ",";
        json << "\"alert_active\":" << (af.drowsiness.alert_active ? "true" : "false") << ",";
        json << "\"drowsy_by_ear\":" << (af.drowsiness.drowsy_by_ear ? "true" : "false") << ",";
        json << "\"drowsy_by_perclos\":" << (af.drowsiness.drowsy_by_perclos ? "true" : "false") << ",";
        json << "\"drowsy_by_yawn\":" << (af.drowsiness.drowsy_by_yawn ? "true" : "false");
        json << "},";
        json << "\"yawn\":{";
        json << "\"is_yawning\":" << (af.yawn.is_yawning_now ? "true" : "false") << ",";
        json << "\"yawn_count_5min\":" << af.yawn.yawn_count_5min;
        json << "}";

        if (include_landmarks && !af.landmarks.empty()) {
            json << ",\"landmarks\":[";
            for (size_t k = 0; k < af.landmarks.size(); ++k) {
                if (k > 0) json << ",";
                json << "[" << af.landmarks[k].x << "," << af.landmarks[k].y << "]";
            }
            json << "]";
        }

        json << "}";
    }

    json << "]";
    json << "}";
    return json.str();
}

bool MqttPublisher::publishResults(uint64_t timestamp_ms, uint32_t frame_id,
                                    const std::vector<AnalyzedFace>& faces,
                                    float inference_time_ms,
                                    bool include_landmarks) {
    std::string payload = buildResultJson(timestamp_ms, frame_id, faces, inference_time_ms, include_landmarks);
    return publish(config_.topic, payload);
}

}  // namespace facemesh_reader
