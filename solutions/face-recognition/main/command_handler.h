#ifndef _FACE_RECOGNITION_COMMAND_HANDLER_H_
#define _FACE_RECOGNITION_COMMAND_HANDLER_H_

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace face_recognition {

enum class CommandOp { Enroll, Remove, List, Reload };
enum class EnrollSource { Camera, Image };

struct Command {
    std::string request_id;
    CommandOp op = CommandOp::List;
    EnrollSource source = EnrollSource::Camera;
    std::string name;
    int frames = 1;                       // camera source: how many to average
    std::vector<uint8_t> encoded_image;   // image source: raw JPEG/PNG bytes
};

// Bounded MQTT command queue.
//
// push() runs on the Mosquitto network thread and does nothing but parse and
// enqueue. Every side effect -- CVI inference, gallery mutation -- happens when
// the frame loop drains the queue, because the NPU is single-core and the frame
// loop owns it.
//
// The queue is capped: a client that floods the command topic must not be able
// to grow this process's heap without bound. Overflow is reported to the caller
// so it can publish a failure ack rather than dropping silently.
class CommandQueue {
public:
    explicit CommandQueue(size_t capacity = 8) : capacity_(capacity) {}

    // Parses one MQTT payload. On success the command is enqueued and true is
    // returned. On failure `error` explains why and `request_id` carries
    // whatever id could be recovered, so the caller can still ack.
    bool push(const std::string& payload, std::string* error, std::string* request_id,
              std::string* op_name);

    bool pop(Command& out);
    size_t size() const;

    static const char* opName(CommandOp op);

private:
    mutable std::mutex mutex_;
    std::deque<Command> queue_;
    size_t capacity_;
};

}  // namespace face_recognition

#endif  // _FACE_RECOGNITION_COMMAND_HANDLER_H_
