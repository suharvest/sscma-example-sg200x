#ifndef _MICRODUCK_HTTP_CONTROL_H_
#define _MICRODUCK_HTTP_CONTROL_H_

#include <atomic>
#include <memory>
#include <thread>

#include "app_config.h"
#include "result_payload.h"

namespace microduck {

class HttpControlServer {
public:
    HttpControlServer();
    ~HttpControlServer();

    bool start(int port, ActionStateMachine* sm, ConfigWatcher* watcher);
    void stop();

private:
    void serverLoop();
    void handleClient(int client_fd);

    int port_ = 8082;
    int server_fd_ = -1;
    ActionStateMachine* sm_ = nullptr;
    ConfigWatcher* watcher_ = nullptr;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> thread_;
};

}  // namespace microduck

#endif  // _MICRODUCK_HTTP_CONTROL_H_
