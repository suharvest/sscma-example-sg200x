#ifndef _YOLO_HTTP_CONTROL_H_
#define _YOLO_HTTP_CONTROL_H_

// Minimal HTTP/1.1 server for the inference control API (see
// inference_control.h). POSIX sockets, one short-lived thread per connection
// so a long-poll on /api/results/latest does not block a toggle.

#include <atomic>
#include <memory>
#include <thread>

#include "inference_control.h"

namespace yolo {

class HttpControlServer {
public:
    static constexpr int kMaxClients = 8;

    explicit HttpControlServer(InferenceControl& ctl) : ctl_(ctl) {}
    ~HttpControlServer();

    /** Listen on 0.0.0.0:@p port. */
    bool start(int port);
    /** Stop accepting, wake long-polls and wait (bounded) for open connections. */
    void stop();

private:
    void accept_loop();
    void handle_client(int fd);

    InferenceControl& ctl_;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    std::atomic<int> clients_{0};
    std::unique_ptr<std::thread> thread_;
};

}  // namespace yolo

#endif  // _YOLO_HTTP_CONTROL_H_
