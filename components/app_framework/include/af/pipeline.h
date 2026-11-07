/**
 * @file pipeline.h
 * @brief Pipeline<State>: a fixed-capacity, allocation-free chain of per-frame steps.
 *
 * A step is a plain function `Flow step(Context<State>&)`. Capture-less lambdas
 * convert to it implicitly; everything a step needs lives in the user-defined
 * State, which is passed by reference and owned by the caller. That keeps the
 * hot path free of std::function, virtual calls and heap allocation.
 *
 * Each step decides what happens next:
 *   - Flow::Next   -- continue with the following step;
 *   - Flow::Reject -- stop early, this frame is not interesting (normal outcome,
 *                     e.g. "no object found, skip pose + classification");
 *   - Flow::Error  -- stop early, something failed.
 *
 * run() returns a RunReport with the outcome, the step that stopped the chain,
 * an optional reason string and the wall time of every executed step (step_us).
 *
 * Example:
 * @code
 *   struct S { Detector* det; int found = 0; };
 *   af::Pipeline<S> p;
 *   p.add("detect", [](af::Context<S>& c) {
 *        c.state.found = c.state.det->run(c.frame);
 *        return c.state.found ? af::Flow::Next : c.reject("no target");
 *    })
 *    .add("classify", [](af::Context<S>& c) { return af::Flow::Next; });
 *   S state{&det};
 *   af::RunReport r = p.run(frame, state);
 * @endcode
 *
 * Header-only template plus a tiny pipeline.cpp (clock, names); no SDK dependency.
 */
#ifndef AF_PIPELINE_H_
#define AF_PIPELINE_H_

#include <cstddef>
#include <cstdint>

#include "af/frame.h"

namespace af {

/** Outcome of one step and of a whole run. */
enum class Flow : uint8_t {
    Next = 0,  ///< continue (for a run: every step returned Next)
    Reject,    ///< stop early, frame not interesting
    Error,     ///< stop early, failure
};

/** "next" / "reject" / "error". */
const char* to_string(Flow f) noexcept;

/** Monotonic clock in microseconds (CLOCK_MONOTONIC). */
uint64_t monotonic_us() noexcept;

/** Clock signature; replaceable for deterministic tests. */
using ClockFn = uint64_t (*)();

/** Maximum steps per pipeline. Fixed so RunReport needs no allocation. */
constexpr size_t kMaxSteps = 16;

/** Value returned by Pipeline::run(). Trivially copyable, ~84 bytes. */
struct RunReport {
    Flow flow = Flow::Next;          ///< Next when all steps completed
    uint8_t executed = 0;            ///< number of steps that ran (including the one that stopped)
    uint8_t stopped_at = 0xFF;       ///< index of the step that returned Reject/Error, 0xFF if none
    const char* reason = nullptr;    ///< set by Context::reject()/error(); static storage expected
    uint32_t total_us = 0;           ///< sum of step_us[0..executed)
    uint32_t step_us[kMaxSteps] = {};  ///< per-step wall time; 0 for steps that did not run

    bool completed() const noexcept { return flow == Flow::Next; }
    bool stopped() const noexcept { return flow != Flow::Next; }
};

/**
 * Per-run context handed to every step.
 *
 * The frame is a copy of the 64-byte view, so a step may narrow it in place
 * (e.g. `c.frame = c.frame.sub_frame(roi)`) for the steps that follow.
 */
template <class State>
struct Context {
    VideoFrame frame;              ///< frame being processed (steps may replace it with a crop)
    State& state;                  ///< user state, owned by the caller
    const char* reason = nullptr;  ///< why the chain stopped; copied into RunReport
    uint8_t step = 0;              ///< index of the currently running step

    Context(const VideoFrame& f, State& s) noexcept : frame(f), state(s) {}

    /** Record @p why and return Flow::Reject. */
    Flow reject(const char* why = nullptr) noexcept {
        reason = why;
        return Flow::Reject;
    }

    /** Record @p why and return Flow::Error. */
    Flow error(const char* why = nullptr) noexcept {
        reason = why;
        return Flow::Error;
    }
};

/**
 * Ordered, fixed-capacity list of steps over a user State.
 *
 * Construction (add()) is expected once at start-up; run() is the per-frame
 * call and performs no allocation. A Pipeline is not thread-safe; run it from
 * one thread (the App's frame loop).
 */
template <class State>
class Pipeline {
public:
    using StepFn = Flow (*)(Context<State>&);

    /**
     * Append a step.
     * @param name static string used in logs and timing reports.
     * @param fn   step function (capture-less lambda or free function).
     * @return *this for chaining. When full or @p fn is null the step is
     *         dropped and ok() turns false, so a misconfigured pipeline is
     *         detectable at start-up instead of silently truncating.
     */
    Pipeline& add(const char* name, StepFn fn) noexcept {
        if (fn == nullptr || count_ >= kMaxSteps) {
            ok_ = false;
            return *this;
        }
        steps_[count_].name = name;
        steps_[count_].fn = fn;
        ++count_;
        return *this;
    }

    /** Replace the clock (tests). nullptr restores monotonic_us. */
    Pipeline& set_clock(ClockFn clock) noexcept {
        clock_ = clock != nullptr ? clock : &monotonic_us;
        return *this;
    }

    /** Run every step in order on @p frame until one does not return Next. */
    RunReport run(const VideoFrame& frame, State& state) const noexcept {
        Context<State> ctx(frame, state);
        return run(ctx);
    }

    /** Same as above with a caller-provided context (lets the caller inspect ctx.frame afterwards). */
    RunReport run(Context<State>& ctx) const noexcept {
        RunReport r;
        uint64_t t0 = clock_();
        for (size_t i = 0; i < count_; ++i) {
            ctx.step = static_cast<uint8_t>(i);
            const Flow f = steps_[i].fn(ctx);
            const uint64_t t1 = clock_();
            r.step_us[i] = static_cast<uint32_t>(t1 - t0);
            r.total_us += r.step_us[i];
            r.executed = static_cast<uint8_t>(i + 1);
            t0 = t1;
            if (f != Flow::Next) {
                r.flow = f;
                r.stopped_at = static_cast<uint8_t>(i);
                r.reason = ctx.reason;
                break;
            }
        }
        return r;
    }

    size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }

    /** False when an add() was dropped (capacity exceeded or null step). */
    bool ok() const noexcept { return ok_; }

    /** Name of step @p i, or nullptr when out of range. */
    const char* name(size_t i) const noexcept { return i < count_ ? steps_[i].name : nullptr; }

private:
    struct Step {
        const char* name = nullptr;
        StepFn fn = nullptr;
    };

    Step steps_[kMaxSteps];
    size_t count_ = 0;
    ClockFn clock_ = &monotonic_us;
    bool ok_ = true;
};

}  // namespace af

#endif  // AF_PIPELINE_H_
