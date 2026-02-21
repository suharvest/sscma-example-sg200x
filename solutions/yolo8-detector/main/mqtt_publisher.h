#ifndef _MQTT_PUBLISHER_H_
#define _MQTT_PUBLISHER_H_

#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

#include <mosquitto.h>

#include "yolo8_detector.h"
#include "person_tracker.h"

namespace yolo8 {

struct MqttConfig {
    std::string host = "localhost";
    int port = 1883;
    std::string username;
    std::string password;
    std::string client_id = "recamera-yolo8-detector";
    std::string topic = "recamera/yolo8/detections";
    int qos = 0;
    bool retain = false;
};

class MqttPublisher {
public:
    MqttPublisher();
    ~MqttPublisher();

    bool init(const MqttConfig& config);
    void deinit();
    bool isConnected() const { return connected_.load(); }

    bool publishResults(uint64_t timestamp_ms, uint32_t frame_id,
                        const std::vector<Detection>& detections,
                        float inference_time_ms);

    bool publishTrackingResults(uint64_t timestamp_ms, uint32_t frame_id,
                                 const std::vector<TrackedPerson>& persons,
                                 const StateCount& counts,
                                 float inference_time_ms);

    bool publish(const std::string& topic, const std::string& payload);

private:
    static void onConnectCallback(struct mosquitto* mosq, void* obj, int rc);
    static void onDisconnectCallback(struct mosquitto* mosq, void* obj, int rc);

    void onConnect(int rc);
    void onDisconnect(int rc);

    std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                                 const std::vector<Detection>& detections,
                                 float inference_time_ms);

    std::string buildTrackingJson(uint64_t timestamp_ms, uint32_t frame_id,
                                   const std::vector<TrackedPerson>& persons,
                                   const StateCount& counts,
                                   float inference_time_ms);

private:
    struct mosquitto* client_;
    MqttConfig config_;
    std::atomic<bool> connected_;
    std::atomic<bool> initialized_;
};

}  // namespace yolo8

#endif  // _MQTT_PUBLISHER_H_
