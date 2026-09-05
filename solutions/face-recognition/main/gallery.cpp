#include "gallery.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "arcface_runner.h"
#include "json.hpp"

namespace face_recognition {

namespace {

constexpr int kFormatVersion = 2;

void setError(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}

std::string nowIso8601() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S+00:00", &tm);
    return std::string(buf);
}

bool makeParentDirs(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return true;
    const std::string dir = path.substr(0, slash);
    std::string acc;
    size_t pos = 0;
    while (pos <= dir.size()) {
        const size_t next = dir.find('/', pos);
        const std::string part = dir.substr(0, next == std::string::npos ? dir.size() : next);
        if (!part.empty()) {
            if (mkdir(part.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return true;
}

}  // namespace

bool Gallery::load(const std::string& path, const std::string& expected_model_tag,
                   float threshold, std::string* error) {
    path_ = path;
    model_tag_ = expected_model_tag;
    threshold_ = threshold;
    return reload(error);
}

bool Gallery::reload(std::string* error) {
    users_.clear();

    std::ifstream in(path_);
    if (!in.is_open()) {
        // A missing gallery is an empty gallery, not a failure: that is the
        // state of every device before the first enrollment.
        return true;
    }

    nlohmann::json blob;
    try {
        in >> blob;
    } catch (const std::exception& e) {
        setError(error, std::string(path_) + ": unparsable JSON: " + e.what());
        return false;
    }
    if (!blob.is_object()) {
        setError(error, path_ + ": top level is not an object");
        return false;
    }

    const auto meta = blob.value("_meta", nlohmann::json::object());
    const std::string tag = meta.value("model_tag", std::string());
    const int dim = meta.value("dim", 0);
    if (tag != model_tag_ || dim != 512) {
        setError(error, path_ + ": gallery is model_tag='" + tag + "' dim=" +
                            std::to_string(dim) + ", running model is '" + model_tag_ +
                            "' dim=512; embeddings are not comparable across models");
        return false;
    }

    const auto users = blob.value("users", nlohmann::json::object());
    if (!users.is_object()) {
        setError(error, path_ + ": 'users' is not an object");
        return false;
    }
    for (auto it = users.begin(); it != users.end(); ++it) {
        const auto& rec = it.value();
        if (!rec.is_object()) continue;
        const auto emb = rec.value("emb", nlohmann::json::array());
        if (!emb.is_array() || emb.size() != 512) {
            setError(error, path_ + ": user '" + it.key() + "' has " +
                                std::to_string(emb.size()) + "-D embedding, expected 512");
            users_.clear();
            return false;
        }
        GalleryRecord g;
        for (size_t i = 0; i < 512; ++i) g.emb[i] = emb[i].get<float>();
        if (!l2Normalize(g.emb)) {
            setError(error, path_ + ": user '" + it.key() + "' has a zero embedding");
            users_.clear();
            return false;
        }
        g.n = rec.value("n", 1u);
        g.ts = rec.value("ts", nowIso8601());
        users_[it.key()] = std::move(g);
    }
    return true;
}

bool Gallery::save(std::string* error) {
    if (path_.empty()) {
        setError(error, "gallery has no path");
        return false;
    }
    if (!makeParentDirs(path_)) {
        setError(error, "cannot create gallery directory for " + path_);
        return false;
    }

    nlohmann::json blob;
    blob["_meta"] = {{"model_tag", model_tag_},
                     {"dim", 512},
                     {"threshold", threshold_},
                     {"updated", nowIso8601()},
                     {"format", kFormatVersion}};
    nlohmann::json users = nlohmann::json::object();
    for (const auto& kv : users_) {
        nlohmann::json emb = nlohmann::json::array();
        for (float v : kv.second.emb) {
            // Six decimals: enough that the round trip costs well under 1e-5 of
            // cosine, and it keeps a 1000-user file readable.
            emb.push_back(std::round(static_cast<double>(v) * 1e6) / 1e6);
        }
        users[kv.first] = {{"emb", emb}, {"n", kv.second.n}, {"ts", kv.second.ts}};
    }
    blob["users"] = users;

    // Atomic replace: a truncated gallery would take every enrolled person with
    // it, and this file is written while the frame loop is the only thing
    // keeping the device useful.
    const std::string tmp = path_ + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            setError(error, "cannot write " + tmp);
            return false;
        }
        out << blob.dump();
        out.flush();
        if (!out.good()) {
            setError(error, "write error on " + tmp);
            out.close();
            ::unlink(tmp.c_str());
            return false;
        }
    }
    if (::rename(tmp.c_str(), path_.c_str()) != 0) {
        setError(error, std::string("rename failed: ") + strerror(errno));
        ::unlink(tmp.c_str());
        return false;
    }
    return true;
}

GalleryMatch Gallery::match(const Embedding512& query) const {
    GalleryMatch m;
    for (const auto& kv : users_) {
        const float s = cosineSimilarity(query, kv.second.emb);
        if (m.name.empty() || s > m.score) {
            m.name = kv.first;
            m.score = s;
        }
    }
    if (m.name.empty()) return GalleryMatch{};
    if (m.score >= threshold_) {
        m.matched = true;
        return m;
    }
    // Keep the near-miss score so a caller can log it, but report no identity.
    const float best = m.score;
    m = GalleryMatch{};
    m.score = best;
    return m;
}

bool Gallery::enroll(const std::string& name, const std::vector<Embedding512>& embeddings,
                     std::string* error) {
    if (name.empty()) {
        setError(error, "enroll: empty name");
        return false;
    }
    if (embeddings.empty()) {
        setError(error, "enroll: no embeddings supplied");
        return false;
    }
    if (users_.find(name) == users_.end() && users_.size() >= maxUsers()) {
        setError(error, "enroll: gallery is full (" + std::to_string(maxUsers()) + " users)");
        return false;
    }

    Embedding512 mean{};
    for (const auto& e : embeddings) {
        Embedding512 tmp = e;
        if (!l2Normalize(tmp)) {
            setError(error, "enroll: zero-norm embedding");
            return false;
        }
        for (size_t i = 0; i < mean.size(); ++i) mean[i] += tmp[i];
    }
    for (auto& v : mean) v /= static_cast<float>(embeddings.size());
    if (!l2Normalize(mean)) {
        setError(error, "enroll: averaged embedding has zero norm");
        return false;
    }

    GalleryRecord rec;
    rec.emb = mean;
    rec.n = static_cast<uint32_t>(embeddings.size());
    rec.ts = nowIso8601();

    auto backup = users_;
    users_[name] = std::move(rec);
    if (!save(error)) {
        users_ = std::move(backup);
        return false;
    }
    return true;
}

bool Gallery::remove(const std::string& name, std::string* error) {
    auto it = users_.find(name);
    if (it == users_.end()) {
        setError(error, "no such user: " + name);
        return false;
    }
    auto backup = users_;
    users_.erase(it);
    if (!save(error)) {
        users_ = std::move(backup);
        return false;
    }
    return true;
}

std::vector<std::pair<std::string, uint32_t>> Gallery::list() const {
    std::vector<std::pair<std::string, uint32_t>> out;
    out.reserve(users_.size());
    for (const auto& kv : users_) out.emplace_back(kv.first, kv.second.n);
    return out;
}

}  // namespace face_recognition
