#ifndef _MICRODUCK_HTTP_CONTROL_H_
#define _MICRODUCK_HTTP_CONTROL_H_

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "app_config.h"
#include "json.hpp"
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
    void handleCalibrateStand(int client_fd, const nlohmann::json& req_json, const nlohmann::json& params);
    void handleSetStandNominal(int client_fd, const nlohmann::json& params);
    bool applyNominal(const AppConfig& next, bool persist, std::string& err);
    static nlohmann::json nominalJson(const AppConfig& c);

    int port_ = 8082;
    int server_fd_ = -1;
    ActionStateMachine* sm_ = nullptr;
    ConfigWatcher* watcher_ = nullptr;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> thread_;
};

}  // namespace microduck

#endif  // _MICRODUCK_HTTP_CONTROL_H_
