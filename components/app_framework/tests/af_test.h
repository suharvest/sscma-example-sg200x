/**
 * @file af_test.h
 * @brief Minimal assertion harness for the host tests (no external framework).
 *
 * TEST(name) registers a function; CHECK/CHECK_EQ record failures and keep
 * going; main() from AF_TEST_MAIN runs everything and returns non-zero on any
 * failure, which is what ctest looks at.
 */
#ifndef AF_TEST_H_
#define AF_TEST_H_

#include <cstdio>

namespace af_test {

struct Case {
    const char* name;
    void (*fn)();
    Case* next;
};

inline Case*& head() {
    static Case* h = nullptr;
    return h;
}

inline int& failures() {
    static int f = 0;
    return f;
}

struct Registrar {
    Case c;
    Registrar(const char* n, void (*fn)()) : c{n, fn, nullptr} {
        Case** p = &head();
        while (*p) p = &(*p)->next;  // keep declaration order
        *p = &c;
    }
};

inline int run_all() {
    int cases = 0;
    for (Case* c = head(); c; c = c->next, ++cases) {
        const int before = failures();
        c->fn();
        std::printf("[%s] %s\n", failures() == before ? " OK " : "FAIL", c->name);
    }
    std::printf("%d cases, %d failed checks\n", cases, failures());
    return failures() == 0 ? 0 : 1;
}

}  // namespace af_test

#define TEST(name)                                                   \
    static void name();                                              \
    static af_test::Registrar name##_reg(#name, &name);              \
    static void name()

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            ++af_test::failures();                                                    \
            std::printf("  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond);    \
        }                                                                             \
    } while (0)

#define CHECK_EQ(a, b)                                                                                              \
    do {                                                                                                            \
        const auto va_ = (a);                                                                                       \
        const auto vb_ = (b);                                                                                       \
        if (!(va_ == vb_)) {                                                                                        \
            ++af_test::failures();                                                                                  \
            std::printf("  %s:%d: CHECK_EQ(%s, %s) failed: %lld vs %lld\n", __FILE__, __LINE__, #a, #b, (long long)va_, \
                        (long long)vb_);                                                                            \
        }                                                                                                           \
    } while (0)

#define AF_TEST_MAIN \
    int main() { return af_test::run_all(); }

#endif  // AF_TEST_H_
