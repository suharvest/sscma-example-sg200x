#ifndef _ARUCO_HTTP_CONTROL_H_
#define _ARUCO_HTTP_CONTROL_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

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
    struct ClientContext {
        int fd = -1;
        std::atomic<bool> done{false};
        std::thread thread;
    };

    void accept_loop();
    void handle_client(int fd);
    void clean_finished_clients();

    InferenceControl& ctl_;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> thread_;
    std::mutex clients_mu_;
    std::vector<std::shared_ptr<ClientContext>> clients_;
};

}  // namespace aruco

#endif  // _ARUCO_HTTP_CONTROL_H_
