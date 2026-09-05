#include "command_handler.h"

#include <algorithm>
#include <cctype>

#include "json.hpp"

namespace face_recognition {

namespace {

const char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

bool base64Decode(const std::string& in, std::vector<uint8_t>& out) {
    int8_t table[256];
    for (int i = 0; i < 256; ++i) table[i] = -1;
    for (int i = 0; i < 64; ++i) table[static_cast<uint8_t>(kB64[i])] = static_cast<int8_t>(i);
    // Accept the URL-safe alphabet too; some clients produce it without saying so.
    table[static_cast<uint8_t>('-')] = 62;
    table[static_cast<uint8_t>('_')] = 63;

    out.clear();
    out.reserve(in.size() * 3 / 4 + 3);
    uint32_t acc = 0;
    int bits = 0;
    for (char ch : in) {
        const uint8_t c = static_cast<uint8_t>(ch);
        if (ch == '=' ) break;
        if (std::isspace(c)) continue;
        const int8_t v = table[c];
        if (v < 0) return false;
        acc = (acc << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFFu));
        }
    }
    return !out.empty();
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

}  // namespace

const char* CommandQueue::opName(CommandOp op) {
    switch (op) {
        case CommandOp::Enroll: return "enroll";
        case CommandOp::Remove: return "remove";
        case CommandOp::List: return "list";
        case CommandOp::Reload: return "reload";
    }
    return "unknown";
}

bool CommandQueue::push(const std::string& payload, std::string* error,
                        std::string* request_id, std::string* op_name) {
    if (request_id) request_id->clear();
    if (op_name) *op_name = "unknown";

    nlohmann::json req;
    try {
        req = nlohmann::json::parse(payload);
    } catch (const std::exception& e) {
        if (error) *error = std::string("bad JSON: ") + e.what();
        return false;
    }
    if (!req.is_object()) {
        if (error) *error = "payload must be a JSON object";
        return false;
    }

    Command cmd;
    if (req.contains("request_id") && req["request_id"].is_string()) {
        cmd.request_id = req["request_id"].get<std::string>();
    }
    if (request_id) *request_id = cmd.request_id;

    const std::string op = lower(trim(req.value("op", std::string())));
    if (op_name) *op_name = op.empty() ? "unknown" : op;

    if (op == "list") {
        cmd.op = CommandOp::List;
    } else if (op == "reload") {
        cmd.op = CommandOp::Reload;
    } else if (op == "remove") {
        cmd.op = CommandOp::Remove;
        cmd.name = trim(req.value("name", std::string()));
        if (cmd.name.empty()) {
            if (error) *error = "missing 'name'";
            return false;
        }
    } else if (op == "enroll") {
        cmd.op = CommandOp::Enroll;
        cmd.name = trim(req.value("name", std::string()));
        if (cmd.name.empty()) {
            if (error) *error = "missing 'name'";
            return false;
        }
        const std::string source = lower(trim(req.value("source", std::string("camera"))));
        if (source == "camera") {
            cmd.source = EnrollSource::Camera;
            const int frames = req.value("frames", 1);
            cmd.frames = std::min(16, std::max(1, frames));
        } else if (source == "image") {
            cmd.source = EnrollSource::Image;
            const std::string b64 = req.value("image_b64", std::string());
            if (b64.empty()) {
                if (error) *error = "source 'image' needs 'image_b64'";
                return false;
            }
            if (!base64Decode(b64, cmd.encoded_image)) {
                if (error) *error = "image_b64 is not valid base64";
                return false;
            }
        } else {
            if (error) *error = "source must be 'camera' or 'image'";
            return false;
        }
    } else {
        if (error) *error = "unknown op; expected enroll, remove, list or reload";
        return false;
    }

    std::lock_guard<std::mutex> lk(mutex_);
    if (queue_.size() >= capacity_) {
        if (error) *error = "command queue is full";
        return false;
    }
    queue_.push_back(std::move(cmd));
    return true;
}

bool CommandQueue::pop(Command& out) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (queue_.empty()) return false;
    out = std::move(queue_.front());
    queue_.pop_front();
    return true;
}

size_t CommandQueue::size() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return queue_.size();
}

}  // namespace face_recognition
