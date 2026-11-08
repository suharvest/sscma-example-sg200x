#ifndef _ARUCO_HTTP_CONTROL_H_
#define _ARUCO_HTTP_CONTROL_H_

#include <atomic>
#include <memory>
#include <thread>

#include "inference_control.h"

namespace aruco {

class HttpControlServer {
public:
    static constexpr int kMaxClients = 8;

    explicit HttpControlServer(InferenceControl& ctl) : ctl_(ctl) {}
    ~HttpControlServer();

    bool start(int port);
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

}  // namespace aruco

#endif  // _ARUCO_HTTP_CONTROL_H_
