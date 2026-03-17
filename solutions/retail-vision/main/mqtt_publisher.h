#ifndef _MQTT_PUBLISHER_H_
#define _MQTT_PUBLISHER_H_

#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

#include <mosquitto.h>

#include "person_tracker.h"
#include "zone_metrics.h"

namespace retail_vision {

struct MqttConfig {
    std::string host = "localhost";
    int port = 1883;
    std::string username;
    std::string password;
    std::string client_id = "recamera-retail-vision";
    std::string topic = "recamera/retail-vision/vision";
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

    // Publish VisionPayload: zone metrics + per-person data
    // frame_width/frame_height: display resolution for absolute pixel coordinate output
    // model_width/model_height: inference input resolution (for letterbox correction)
    bool publishVisionPayload(uint64_t timestamp_ms, uint32_t frame_id,
                               float fps, float inference_time_ms,
                               const ZoneSnapshot& zone,
                               const std::vector<TrackedPerson>& persons,
                               int frame_width, int frame_height,
                               int model_width, int model_height);

private:
    static void onConnectCallback(struct mosquitto* mosq, void* obj, int rc);
    static void onDisconnectCallback(struct mosquitto* mosq, void* obj, int rc);
    void onConnect(int rc);
    void onDisconnect(int rc);

    bool publish(const std::string& topic, const std::string& payload);

    std::string buildVisionJson(uint64_t timestamp_ms, uint32_t frame_id,
                                 float fps, float inference_time_ms,
                                 const ZoneSnapshot& zone,
                                 const std::vector<TrackedPerson>& persons,
                                 int frame_width, int frame_height,
                                 int model_width, int model_height);

    struct mosquitto* client_;
    MqttConfig config_;
    std::atomic<bool> connected_;
    std::atomic<bool> initialized_;
};

}  // namespace retail_vision

#endif  // _MQTT_PUBLISHER_H_
