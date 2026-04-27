#ifndef _MQTT_PUBLISHER_H_
#define _MQTT_PUBLISHER_H_

#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

#include <mosquitto.h>

namespace facemesh_reader {

// Forward decl from facemesh_pipeline.h to avoid circular include.
struct AnalyzedFace;

struct MqttConfig {
    std::string host = "localhost";
    int port = 1883;
    std::string username;
    std::string password;
    std::string client_id = "recamera-facemesh-reader";
    std::string topic = "recamera/facemesh-reader/results";
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

    // Publish FaceMesh results (EAR / MAR / drowsiness flags).
    // include_landmarks: if true, embed all 468 (x,y) pixel-coordinate landmarks per face.
    bool publishResults(uint64_t timestamp_ms, uint32_t frame_id,
                        const std::vector<AnalyzedFace>& faces,
                        float inference_time_ms,
                        bool include_landmarks = false);

    bool publish(const std::string& topic, const std::string& payload);

private:
    static void onConnectCallback(struct mosquitto* mosq, void* obj, int rc);
    static void onDisconnectCallback(struct mosquitto* mosq, void* obj, int rc);

    void onConnect(int rc);
    void onDisconnect(int rc);

    std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                                 const std::vector<AnalyzedFace>& faces,
                                 float inference_time_ms,
                                 bool include_landmarks);

private:
    struct mosquitto* client_;
    MqttConfig config_;
    std::atomic<bool> connected_;
    std::atomic<bool> initialized_;
};

}  // namespace facemesh_reader

#endif  // _MQTT_PUBLISHER_H_
