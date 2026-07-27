#include "mp.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int checked = 0;
// 跑一遍完整遍历；只要不崩、不死循环、不越界即算通过。
// 所有返回的 span 必须落在 body 范围内——这是最关键的不变量。
static void walk(const std::string& body, const std::string& ct, int max_iter = 10000) {
    const std::string b = multipart_boundary(ct);
    size_t pos = 0; int guard = 0;
    MultipartSpan s;
    while (true) {
        size_t next = multipart_next(body, b, pos, &s);
        if (next == 0) break;
        // 不变量：span 必须完全在 body 内
        assert(s.data_off <= body.size());
        assert(s.data_len <= body.size());
        assert(s.data_off + s.data_len <= body.size());
        // 不变量：必须前进，否则死循环
        assert(next > pos);
        pos = next;
        if (++guard > max_iter) { printf("FAIL: 未收敛\n"); abort(); }
    }
    checked++;
}

int main() {
    // ---- 定向畸形用例 ----
    struct { const char* name; const char* ct; std::string body; } cases[] = {
      {"Content-Type 无 boundary", "multipart/form-data", "--B\r\n\r\nx\r\n--B--\r\n"},
      {"boundary 为空",            "multipart/form-data; boundary=", "--\r\n\r\nx\r\n----\r\n"},
      {"body 里没有 boundary",     "multipart/form-data; boundary=B", "完全无关的内容"},
      {"body 为空",                "multipart/form-data; boundary=B", ""},
      {"只有开始分隔符",           "multipart/form-data; boundary=B", "--B"},
      {"缺 CRLFCRLF 头分隔",       "multipart/form-data; boundary=B", "--B\r\nContent-Disposition: x\r\n--B--"},
      {"截断：无结束分隔符",       "multipart/form-data; boundary=B", "--B\r\n\r\ndata"},
      {"空 part",                  "multipart/form-data; boundary=B", "--B\r\n\r\n\r\n--B--\r\n"},
      {"数据里含 boundary 片段",   "multipart/form-data; boundary=B", "--B\r\n\r\naa--Bxx\r\n--B--\r\n"},
      {"带引号 boundary",          "multipart/form-data; boundary=\"B\"", "--B\r\n\r\nx\r\n--B--\r\n"},
      {"引号未闭合",               "multipart/form-data; boundary=\"B",  "--B\r\n\r\nx\r\n--B--\r\n"},
      {"boundary 后带参数",        "multipart/form-data; boundary=B; charset=utf-8", "--B\r\n\r\nx\r\n--B--\r\n"},
      {"无 name/filename",         "multipart/form-data; boundary=B", "--B\r\nX: 1\r\n\r\nd\r\n--B--\r\n"},
      {"name 引号未闭合",          "multipart/form-data; boundary=B", "--B\r\nContent-Disposition: form-data; name=\"a\r\n\r\nd\r\n--B--\r\n"},
      {"超长 header",              "multipart/form-data; boundary=B", "--B\r\n" + std::string(100000,'H') + "\r\n\r\nd\r\n--B--\r\n"},
      {"仅结束分隔符",             "multipart/form-data; boundary=B", "--B--\r\n"},
      {"CRLF 缺失",                "multipart/form-data; boundary=B", "--B\n\nd\n--B--"},
      {"NUL 字节",                 "multipart/form-data; boundary=B", std::string("--B\r\n\r\nab\0cd\r\n--B--\r\n", 21)},
      {"多个 part",                "multipart/form-data; boundary=B", "--B\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\n1\r\n--B\r\nContent-Disposition: form-data; name=\"b\"\r\n\r\n2\r\n--B--\r\n"},
      {"boundary 含正则字符",      "multipart/form-data; boundary=.*+?", "--.*+?\r\n\r\nx\r\n--.*+?--\r\n"},
    };
    for (auto& c : cases) { walk(c.body, c.ct); printf("  ok  %s\n", c.name); }

    // ---- 正确性：多 part 必须解析出正确内容 ----
    {
        std::string ct = "multipart/form-data; boundary=B";
        std::string body = "--B\r\nContent-Disposition: form-data; name=\"a\"; filename=\"f1\"\r\n\r\nHELLO\r\n"
                           "--B\r\nContent-Disposition: form-data; name=\"b\"\r\n\r\nWORLD\r\n--B--\r\n";
        std::string bnd = multipart_boundary(ct);
        MultipartSpan s; size_t pos = 0; std::vector<std::string> got, names;
        while ((pos = multipart_next(body, bnd, pos, &s)) != 0) {
            got.push_back(body.substr(s.data_off, s.data_len));
            names.push_back(s.name + "/" + s.filename);
        }
        assert(got.size() == 2);
        assert(got[0] == "HELLO"); assert(got[1] == "WORLD");
        assert(names[0] == "a/f1"); assert(names[1] == "b/");
        printf("  ok  多 part 内容与 name/filename 正确\n");
    }

    // ---- 随机模糊 ----
    std::mt19937 rng(12345);
    const char* alpha = "-\r\n\"=;BdataContent-Disposition:form;name=filename\0 \t";
    const size_t alen = 50;
    for (int i = 0; i < 200000; ++i) {
        size_t n = rng() % 400;
        std::string body; body.reserve(n);
        for (size_t k = 0; k < n; ++k) body.push_back(alpha[rng() % alen]);
        static const char* cts[] = {"multipart/form-data; boundary=B",
                                    "multipart/form-data; boundary=",
                                    "multipart/form-data",
                                    "multipart/form-data; boundary=\"B\"",
                                    "text/plain"};
        walk(body, cts[rng() % 5]);
    }
    // ---- 变异已知合法输入 ----
    std::string base = "--B\r\nContent-Disposition: form-data; name=\"a\"; filename=\"f\"\r\n\r\nPAYLOAD\r\n--B--\r\n";
    for (int i = 0; i < 200000; ++i) {
        std::string m = base;
        int muts = 1 + rng() % 4;
        for (int k = 0; k < muts; ++k) {
            if (m.empty()) break;
            switch (rng() % 3) {
            case 0: m[rng() % m.size()] = static_cast<char>(rng() % 256); break;
            case 1: m.erase(rng() % m.size(), 1 + rng() % 5); break;
            case 2: m.insert(rng() % m.size(), 1 + rng() % 5, static_cast<char>(rng() % 256)); break;
            }
        }
        walk(m, "multipart/form-data; boundary=B");
    }
    printf("\n%d 例全部通过：无崩溃、无越界、无死循环\n", checked);
    return 0;
}
