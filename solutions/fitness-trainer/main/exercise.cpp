#include "exercise.h"

#include <algorithm>
#include <cmath>

namespace fitness {

// ---------------------------------------------------------------------------
// RepCounter
// ---------------------------------------------------------------------------

RepCounter::RepCounter(float up_threshold, float down_threshold, float min_interval_sec)
    : up_threshold_(up_threshold), down_threshold_(down_threshold), min_interval_(min_interval_sec) {}

void RepCounter::reset() {
    phase_ = Phase::Unknown;
    has_smoothed_ = false;
    smoothed_ = 0.0f;
    rep_min_ = 180.0f;
    last_rep_min_ = 180.0f;
    last_rep_time_ = -1e9;
    last_reading_time_ = -1e9;
    // ever_read_ deliberately survives reset(): it describes the camera view
    // (was this side ever visible), not the workout.
}

void RepCounter::miss(double now_sec) {
    if (has_smoothed_ && now_sec - last_reading_time_ > kLostSeconds) {
        phase_ = Phase::Unknown;
        has_smoothed_ = false;
        rep_min_ = 180.0f;
    }
}

bool RepCounter::update(float angle, double now_sec) {
    if (!isReading(angle)) {
        miss(now_sec);
        return false;
    }

    ever_read_ = true;
    last_reading_time_ = now_sec;
    smoothed_ = has_smoothed_ ? (kSmoothingAlpha * angle + (1.0f - kSmoothingAlpha) * smoothed_) : angle;
    has_smoothed_ = true;

    if (phase_ == Phase::Flexed) {
        rep_min_ = std::min(rep_min_, smoothed_);
    }

    if (smoothed_ < down_threshold_) {
        if (phase_ != Phase::Flexed) {
            phase_ = Phase::Flexed;
            rep_min_ = smoothed_;
        }
        return false;
    }

    if (smoothed_ > up_threshold_) {
        const bool completes = (phase_ == Phase::Flexed);
        phase_ = Phase::Extended;
        if (completes) {
            // Debounce: a genuine rep cannot be faster than min_interval_.
            // Anything quicker is the angle rattling across both thresholds.
            if (now_sec - last_rep_time_ < min_interval_) {
                return false;
            }
            last_rep_time_ = now_sec;
            last_rep_min_ = rep_min_;
            rep_min_ = 180.0f;
            return true;
        }
        return false;
    }

    // Inside the hysteresis band: hold the current phase.
    return false;
}

// ---------------------------------------------------------------------------
// Exercise base
// ---------------------------------------------------------------------------

void Exercise::setTargets(int target_reps, int target_sets) {
    target_reps_ = std::max(1, target_reps);
    target_sets_ = std::max(1, target_sets);
}

void Exercise::reset() {
    const bool two_sided = state_.two_sided;
    state_ = ExerciseState{};
    state_.two_sided = two_sided;
    onReset();
}

void Exercise::update(const Pose* pose, double now_sec) {
    state_.rep_completed = false;
    state_.set_completed = false;
    state_.form_warning.clear();

    if (pose == nullptr || pose->empty()) {
        state_.tracking = false;
        state_.has_angle = false;
        state_.stage = "idle";
        // Still tick the counters so a departed athlete's phase decays.
        track(Pose{}, now_sec);
        return;
    }

    const int completed = track(*pose, now_sec);
    if (completed <= 0 || state_.workout_complete) {
        return;
    }

    for (int i = 0; i < completed; ++i) {
        state_.reps++;
        state_.rep_completed = true;
        if (state_.reps >= target_reps_) {
            state_.set_completed = true;
            if (state_.set >= target_sets_) {
                state_.workout_complete = true;
                state_.reps = target_reps_;
                break;
            }
            state_.set++;
            state_.reps = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Squat -- knee flexion (hip / knee / ankle)
// ---------------------------------------------------------------------------
//
// The original measured the torso-thigh angle (shoulder / hip / knee). Knee
// flexion is the standard depth metric and, more practically, it survives the
// athlete facing the camera: the torso angle barely changes head-on, so a
// front-facing squat counted nothing.

namespace {

constexpr float kSquatUp = 160.0f;        // standing
constexpr float kSquatDown = 100.0f;      // parallel-ish
constexpr float kSquatPartial = 120.0f;   // shallower than this = partial rep

constexpr float kPushUpUp = 150.0f;       // arms locked out
constexpr float kPushUpDown = 95.0f;      // chest down
constexpr float kPushUpPartial = 110.0f;

constexpr float kCurlExtended = 150.0f;   // arm hanging
constexpr float kCurlFlexed = 50.0f;      // fully curled
constexpr float kCurlElbowDrift = 40.0f;  // upper arm vs torso, degrees

class Squat : public Exercise {
public:
    const char* id() const override { return "squat"; }
    const char* displayName() const override { return "Squat"; }

protected:
    void onReset() override { counter_.reset(); }

    int track(const Pose& pose, double now_sec) override {
        if (pose.empty()) {
            counter_.miss(now_sec);
            return 0;
        }

        const float left = pose.sideScore({Joint::LeftHip, Joint::LeftKnee, Joint::LeftAnkle});
        const float right = pose.sideScore({Joint::RightHip, Joint::RightKnee, Joint::RightAnkle});
        if (left <= 0.0f && right <= 0.0f) {
            state_.tracking = false;
            state_.has_angle = false;
            state_.stage = "out of frame";
            counter_.miss(now_sec);
            return 0;
        }

        const bool use_left = left >= right;
        const float angle = jointAngle(pose.at(use_left ? Joint::LeftHip : Joint::RightHip),
                                       pose.at(use_left ? Joint::LeftKnee : Joint::RightKnee),
                                       pose.at(use_left ? Joint::LeftAnkle : Joint::RightAnkle));

        const bool rep = counter_.update(angle, now_sec);

        state_.tracking = counter_.hasReading();
        state_.has_angle = state_.tracking;
        state_.angle = counter_.smoothed();
        state_.stage = counter_.flexed() ? "down" : (counter_.extended() ? "up" : "idle");

        if (rep && counter_.lastRepMinAngle() > kSquatPartial) {
            state_.form_warning = "Partial rep - squat deeper";
        }
        return rep ? 1 : 0;
    }

private:
    RepCounter counter_{kSquatUp, kSquatDown, 0.5f};
};

// ---------------------------------------------------------------------------
// Push-up -- elbow flexion (shoulder / elbow / wrist)
// ---------------------------------------------------------------------------

class PushUp : public Exercise {
public:
    const char* id() const override { return "push_up"; }
    const char* displayName() const override { return "Push-up"; }

protected:
    void onReset() override { counter_.reset(); }

    int track(const Pose& pose, double now_sec) override {
        if (pose.empty()) {
            counter_.miss(now_sec);
            return 0;
        }

        const float left = pose.sideScore({Joint::LeftShoulder, Joint::LeftElbow, Joint::LeftWrist});
        const float right = pose.sideScore({Joint::RightShoulder, Joint::RightElbow, Joint::RightWrist});
        if (left <= 0.0f && right <= 0.0f) {
            state_.tracking = false;
            state_.has_angle = false;
            state_.stage = "out of frame";
            counter_.miss(now_sec);
            return 0;
        }

        const bool use_left = left >= right;
        const float angle = jointAngle(pose.at(use_left ? Joint::LeftShoulder : Joint::RightShoulder),
                                       pose.at(use_left ? Joint::LeftElbow : Joint::RightElbow),
                                       pose.at(use_left ? Joint::LeftWrist : Joint::RightWrist));

        const bool rep = counter_.update(angle, now_sec);

        state_.tracking = counter_.hasReading();
        state_.has_angle = state_.tracking;
        state_.angle = counter_.smoothed();
        state_.stage = counter_.flexed() ? "down" : (counter_.extended() ? "up" : "idle");

        if (rep && counter_.lastRepMinAngle() > kPushUpPartial) {
            state_.form_warning = "Partial rep - lower your chest";
        }
        return rep ? 1 : 0;
    }

private:
    // 1.0s in the original, which rejected fast-but-real reps; the hysteresis
    // band now does most of the de-bouncing, so the floor can come down.
    RepCounter counter_{kPushUpUp, kPushUpDown, 0.5f};
};

// ---------------------------------------------------------------------------
// Hammer curl -- both arms, counted independently
// ---------------------------------------------------------------------------
//
// A "rep" for the set counter is one curl on BOTH arms when both are visible
// (so alternating curls advance at the pace of the slower arm), or on the one
// visible arm when only one is in view -- otherwise a single-arm workout, or a
// camera that only ever sees one side, would sit at zero forever.

class HammerCurl : public Exercise {
public:
    HammerCurl() { state_.two_sided = true; }

    const char* id() const override { return "hammer_curl"; }
    const char* displayName() const override { return "Hammer Curl"; }

protected:
    void onReset() override {
        left_.reset();
        right_.reset();
        paired_ = 0;
    }

    int track(const Pose& pose, double now_sec) override {
        if (pose.empty()) {
            left_.miss(now_sec);
            right_.miss(now_sec);
            return 0;
        }

        const bool has_left = pose.sideScore({Joint::LeftShoulder, Joint::LeftElbow, Joint::LeftWrist}) > 0.0f;
        const bool has_right = pose.sideScore({Joint::RightShoulder, Joint::RightElbow, Joint::RightWrist}) > 0.0f;

        if (!has_left && !has_right) {
            state_.tracking = false;
            state_.has_angle = false;
            state_.stage = "out of frame";
            left_.miss(now_sec);
            right_.miss(now_sec);
            return 0;
        }

        const float angle_left = has_left ? jointAngle(pose.at(Joint::LeftShoulder),
                                                       pose.at(Joint::LeftElbow),
                                                       pose.at(Joint::LeftWrist))
                                          : std::nanf("");
        const float angle_right = has_right ? jointAngle(pose.at(Joint::RightShoulder),
                                                         pose.at(Joint::RightElbow),
                                                         pose.at(Joint::RightWrist))
                                            : std::nanf("");

        if (left_.update(angle_left, now_sec)) state_.reps_left++;
        if (right_.update(angle_right, now_sec)) state_.reps_right++;

        state_.tracking = left_.hasReading() || right_.hasReading();
        // The primary angle is whichever arm is further into the curl, so a
        // single number still reflects what the athlete is doing.
        if (left_.hasReading() && right_.hasReading()) {
            state_.angle = std::min(left_.smoothed(), right_.smoothed());
        } else if (left_.hasReading()) {
            state_.angle = left_.smoothed();
        } else if (right_.hasReading()) {
            state_.angle = right_.smoothed();
        }
        state_.has_angle = state_.tracking;

        const bool curling = left_.flexed() || right_.flexed();
        state_.stage = curling ? "curl" : (state_.tracking ? "extend" : "idle");

        // Form: upper arm should stay against the torso. Angle at the shoulder
        // between the elbow and the hip -- the original's check, kept as-is.
        checkElbowDrift(pose, has_left, has_right);

        const int paired_now = pairedReps();
        const int gained = paired_now - paired_;
        paired_ = paired_now;
        return std::max(0, gained);
    }

private:
    void checkElbowDrift(const Pose& pose, bool has_left, bool has_right) {
        if (has_left && pose.visible(Joint::LeftHip)) {
            const float drift = jointAngle(pose.at(Joint::LeftElbow), pose.at(Joint::LeftShoulder),
                                           pose.at(Joint::LeftHip));
            if (isReading(drift) && drift > kCurlElbowDrift) {
                state_.form_warning = "Left elbow drifting - keep it at your side";
                return;
            }
        }
        if (has_right && pose.visible(Joint::RightHip)) {
            const float drift = jointAngle(pose.at(Joint::RightElbow), pose.at(Joint::RightShoulder),
                                           pose.at(Joint::RightHip));
            if (isReading(drift) && drift > kCurlElbowDrift) {
                state_.form_warning = "Right elbow drifting - keep it at your side";
            }
        }
    }

    int pairedReps() const {
        const bool seen_left = left_.everRead();
        const bool seen_right = right_.everRead();
        if (seen_left && seen_right) {
            return std::min(state_.reps_left, state_.reps_right);
        }
        return std::max(state_.reps_left, state_.reps_right);
    }

    RepCounter left_{kCurlExtended, kCurlFlexed, 0.4f};
    RepCounter right_{kCurlExtended, kCurlFlexed, 0.4f};
    int paired_ = 0;
};

}  // namespace

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

std::unique_ptr<Exercise> Exercise::create(const std::string& id) {
    if (id == "squat") return std::unique_ptr<Exercise>(new Squat());
    if (id == "push_up") return std::unique_ptr<Exercise>(new PushUp());
    if (id == "hammer_curl") return std::unique_ptr<Exercise>(new HammerCurl());
    return nullptr;
}

std::vector<std::string> Exercise::ids() {
    return {"squat", "push_up", "hammer_curl"};
}

bool Exercise::known(const std::string& id) {
    const auto all = ids();
    return std::find(all.begin(), all.end(), id) != all.end();
}

}  // namespace fitness
