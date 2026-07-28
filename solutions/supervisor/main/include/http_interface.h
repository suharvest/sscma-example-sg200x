#ifndef _HTTP_INTERFACE_H_
#define _HTTP_INTERFACE_H_

#include <ctime>
#include <string>
#include <unordered_map>
#include <vector>

#include "http_request.h"
#include "json.hpp"
#include "logger.hpp"

using json = nlohmann::json;

typedef enum {
    API_STATUS_OK = 0,
    API_STATUS_NEXT,
    API_STATUS_ERROR,
    API_STATUS_REPLY_FILE,
    API_STATUS_AUTHORIZED,
    API_STATUS_UNAUTHORIZED,
    // #14: the handler enqueued a long operation on the worker pool and the
    // reply will be sent later, once the worker finishes. The dispatcher must
    // keep the connection open and NOT reply now.
    API_STATUS_ASYNC,
} api_status_t;
typedef const struct http_request* request_t;
typedef json& response_t;
typedef api_status_t (*api_handler_t)(request_t req, response_t res);

class http_interface {
public:
    http_interface() = default;
    virtual ~http_interface() = default;

    // #14: id of the mongoose connection whose MG_EV_HTTP_MSG is being
    // dispatched right now. Set by http_server on the poll thread just before
    // it calls api_base::api_handler(), so an async handler can remember which
    // connection to reply to once its worker finishes. All handlers run on the
    // single poll thread, so a plain static is race-free here.
    static void set_conn_id(unsigned long id) { _conn_id = id; }
    static unsigned long conn_id() { return _conn_id; }

protected:
    static inline const std::string STR_OK = "OK";
    static inline const std::string STR_FAILED = "Failed";

    template <typename T>
    static json to_json(T&& obj)
    {
        json j = json::object();
        try {
            j = json::parse(obj);
        } catch (const std::exception& e) {
            LOGV("%s", e.what());
        }
        return j;
    }

    static void response(response_t res, int code = 0,
        const std::string& msg = STR_OK, const json& data = json::object())
    {
        res["code"] = code;
        res["msg"] = msg;
        res["data"] = data;
    }

    static std::string get_uri(request_t req)
    {
        return std::string(req->uri);
    }

    static std::string get_host(request_t req, bool ip_only = true)
    {
        std::string host = _get_header_var(req, "Host");
        if (ip_only) {
            size_t pos = host.find(':');
            if (pos != std::string::npos) {
                host = host.substr(0, pos);
            }
        }
        return host;
    }

    static std::string get_port(request_t req)
    {
        std::string hdr = _get_header_var(req, "Host");
        size_t pos = hdr.find(':');
        return (pos != std::string::npos && pos < hdr.length() - 1)
            ? hdr.substr(pos + 1)
            : "";
    }

    static std::string get_param(request_t req, std::string param)
    {
        auto&& val = _get_http_var(req, param);
        if (!val.empty()) {
            return val;
        }
        return _get_header_var(req, param.c_str());
    }

    static std::string get_body_raw(request_t req)
    {
        return std::string(req->body);
    }

    static json parse_body(request_t req)
    {
        std::string type = _get_header_var(req, "Content-Type");
        if (type.find("application/json") == std::string::npos) {
            return json({});
        }
        return to_json(get_body_raw(req));
    }

    typedef struct {
        std::string name;
        std::string filename;
        char* data;
        size_t len;
    } multipart_t;
    static std::vector<multipart_t> get_multiparts(request_t req, const std::string& param = "")
    {
        std::vector<multipart_t> parts;
        auto&& type = _get_header_var(req, "Content-Type");
        if (type.find("multipart/form-data") != std::string::npos) {
            size_t pos = 0;
            http_multipart_part part;
            while (req->next_multipart != nullptr
                && (pos = req->next_multipart(req, pos, &part)) > 0) {
                multipart_t mp;
                mp.name = std::string(part.name);
                mp.filename = std::string(part.filename);
                mp.data = part.data;
                mp.len = part.len;
                parts.emplace_back(mp);
            }
        } else {
            if (!param.empty()) {
                multipart_t mp;
                mp.name = param;
                mp.filename = get_param(req, param);
                // const_cast: multipart_t::data is non-const for historical
                // reasons; nothing writes through it.
                mp.data = const_cast<char*>(req->body.data());
                mp.len = req->body.size();
                parts.emplace_back(mp);
            }
        }

        return parts;
    }

    // token
    static constexpr uint32_t TOKEN_EXPIRATION_TIME = 3 * 60 * 60 * 24; // 3 days

    /* Session age is measured on CLOCK_MONOTONIC, never on the wall clock.
     * reCamera has no battery-backed RTC, so it boots somewhere near the epoch
     * and the first thing a user does is push the browser's clock onto the
     * device (Device -> sync from browser). With time(nullptr) that decades-wide
     * jump makes every live session instantly older than TOKEN_EXPIRATION_TIME:
     * the very next request after set_timestamp 401s and the user is bounced to
     * the login page mid-flow. CLOCK_MONOTONIC is unaffected by settimeofday(),
     * so a clock correction no longer kills the session that performed it.
     * Trade-off: the clock resets across reboot, but so does _tokens (it is
     * process-local in-memory state), so the two stay consistent. */
    static time_t _mono_now()
    {
        struct timespec ts;
        if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
            return time(nullptr); // should not happen; degrade to old behaviour
        }
        return static_cast<time_t>(ts.tv_sec);
    }

    static void save_token(std::string& token)
    {
        _tokens[token] = _mono_now();
        LOGV("save_token: %s, mono: %ld", token.c_str(), _tokens[token]);
    }

    static bool check_token(std::string& token)
    {
        if (_tokens.find(token) == _tokens.end()) {
            LOGE("Not found token");
            return false;
        }
        if (_tokens[token] + TOKEN_EXPIRATION_TIME < _mono_now()) {
            LOGV("Expired token");
            _tokens.erase(token);
            return false;
        }
        LOGV("Valid token");
        return true;
    }

private:
    static inline unsigned long _conn_id = 0;
    static inline std::unordered_map<std::string, time_t> _tokens;

    static std::string _get_http_var(request_t req, std::string param)
    {
        if (req == nullptr || req->query_var == nullptr) {
            return "";
        }
        return std::string(req->query_var(req, param.c_str()));
    }

    static std::string _get_header_var(request_t req, const char* name)
    {
        if (req == nullptr || req->header == nullptr) {
            return "";
        }
        return std::string(req->header(req, name));
    }
};

#endif // _HTTP_INTERFACE_H_
