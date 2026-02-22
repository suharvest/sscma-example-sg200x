#ifndef _MQTT_PUBLISHER_H_
#define _MQTT_PUBLISHER_H_

#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

#include <mosquitto.h>

#include "detector.h"

namespace detection_blur {

struct MqttConfig {
    std::string host = "localhost";
    int port = 1883;
    std::string username;
    std::string password;
    std::string client_id = "recamera-detection-blur";
    std::string topic = "recamera/detection-blur/results";
    int qos = 0;
    bool retain = false;
};

class MqttPublisher {
public:
    MqttPublisher();
    ~MqttPublisher();

    // Initialize and connect to MQTT broker
    bool init(const MqttConfig& config);

    // Disconnect and cleanup
    void deinit();

    // Check if connected
    bool isConnected() const { return connected_.load(); }

    // Publish detection results
    bool publishResults(uint64_t timestamp_ms, uint32_t frame_id,
                        const std::vector<DetectionBox>& detections,
                        float inference_time_ms);

    // Publish raw JSON message
    bool publish(const std::string& topic, const std::string& payload);

private:
    static void onConnectCallback(struct mosquitto* mosq, void* obj, int rc);
    static void onDisconnectCallback(struct mosquitto* mosq, void* obj, int rc);

    void onConnect(int rc);
    void onDisconnect(int rc);

    std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                                 const std::vector<DetectionBox>& detections,
                                 float inference_time_ms);

private:
    struct mosquitto* client_;
    MqttConfig config_;
    std::atomic<bool> connected_;
    std::atomic<bool> initialized_;
};

}  // namespace detection_blur

#endif  // _MQTT_PUBLISHER_H_
