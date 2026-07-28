#ifndef _FITNESS_EXERCISE_H_
#define _FITNESS_EXERCISE_H_

// Exercise tracking: turn a stream of joint angles into a rep count.
//
// Ported from the Python original's exercises/{squat,push_up,hammer_curl}.py.
// The thresholds and the general shape of the state machine are the original's;
// three things are deliberately different, because a webcam on a laptop and an
// INT8 model on a TPU do not fail the same way:
//
//   1. A rep is counted on the way BACK UP, not at the bottom. The original
//      incremented on entering the bottom position, so an athlete who dropped
//      into a squat and stood back up halfway scored the same as one who
//      completed it. Counting the return also makes the "partial rep" check
//      below possible, since the full range of the rep is known by then.
//   2. Hysteresis with an explicit debounce, not a chain of if/elif over a
//      free-form stage string. Quantised keypoints jitter by a few degrees;
//      a single threshold crossed twice in consecutive frames counted two reps.
//   3. Both sides are read and the better-facing one drives the count. The
//      original hard-coded one side, which silently stopped counting when the
//      athlete turned around.
//
// Adding a fourth exercise: subclass Exercise, implement track(), and add one
// line to the table in Exercise::create() plus one enum option in the manifest's
// config_schema. Nothing else knows the list.

#include <memory>
#include <string>
#include <vector>

#include "pose.h"

namespace fitness {

// What the rest of the app reads each frame.
struct ExerciseState {
    int reps = 0;   // reps completed in the CURRENT set
    int set = 1;    // 1-based
    bool workout_complete = false;

    bool tracking = false;         // a usable reading came out of this frame
    std::string stage = "idle";    // exercise-specific, user-facing
    float angle = 0.0f;            // primary tracked angle, degrees
    bool has_angle = false;

    // Two-sided exercises (hammer curl) report each arm; single-sided ones
    // leave two_sided false and these zeroed.
    bool two_sided = false;
    int reps_left = 0;
    int reps_right = 0;

    std::string form_warning;      // empty when form is fine

    // Edge flags, true only on the frame the event happened.
    bool rep_completed = false;
    bool set_completed = false;
};

// Hysteresis rep counter shared by every exercise.
//
// Phase is "extended" above up_threshold and "flexed" below down_threshold,
// with the band between them holding the previous phase -- that band is what
// stops keypoint jitter from counting reps. A rep completes on
// flexed -> extended, no sooner than min_interval_sec after the last one.
class RepCounter {
public:
    RepCounter(float up_threshold, float down_threshold, float min_interval_sec = 0.4f);

    // Feed one angle reading. Returns true on the frame a rep completes.
    bool update(float angle, double now_sec);
    // Feed a frame with no usable reading (joints hidden). After
    // kLostSeconds of these the phase resets, so an athlete who walks away
    // and comes back does not resume mid-rep.
    void miss(double now_sec);

    void reset();

    bool extended() const { return phase_ == Phase::Extended; }
    bool flexed() const { return phase_ == Phase::Flexed; }
    bool idle() const { return phase_ == Phase::Unknown; }

    float smoothed() const { return smoothed_; }
    bool hasReading() const { return has_smoothed_; }

    // Deepest (smallest) angle reached during the rep that just completed.
    // Only meaningful on the frame update() returned true.
    float lastRepMinAngle() const { return last_rep_min_; }

    // Whether this counter has ever produced a reading -- lets a two-sided
    // exercise tell "arm at zero reps" from "arm never seen".
    bool everRead() const { return ever_read_; }

private:
    enum class Phase { Unknown, Extended, Flexed };

    static constexpr float kSmoothingAlpha = 0.5f;  // EMA over the raw angle
    static constexpr double kLostSeconds = 1.5;     // miss() budget before reset

    float up_threshold_;
    float down_threshold_;
    double min_interval_;

    Phase phase_ = Phase::Unknown;
    float smoothed_ = 0.0f;
    bool has_smoothed_ = false;
    bool ever_read_ = false;
    float rep_min_ = 180.0f;
    float last_rep_min_ = 180.0f;
    double last_rep_time_ = -1e9;
    double last_reading_time_ = -1e9;
};

class Exercise {
public:
    virtual ~Exercise() = default;

    virtual const char* id() const = 0;
    virtual const char* displayName() const = 0;

    // Advance the state machine by one frame. Pass pose == nullptr when nobody
    // was detected.
    void update(const Pose* pose, double now_sec);

    void reset();
    void setTargets(int target_reps, int target_sets);

    const ExerciseState& state() const { return state_; }
    int targetReps() const { return target_reps_; }
    int targetSets() const { return target_sets_; }

    // The registry. ids() drives nothing at runtime but keeps the manifest's
    // enum options and the code in one place when the list grows.
    static std::unique_ptr<Exercise> create(const std::string& id);
    static std::vector<std::string> ids();
    static bool known(const std::string& id);

protected:
    // Per-exercise tracking. Fill in state_.stage / angle / form_warning and
    // return the number of reps completed on this frame (0 or 1; hammer curl
    // may return 1 when either arm completes).
    virtual int track(const Pose& pose, double now_sec) = 0;
    virtual void onReset() = 0;

    ExerciseState state_;
    int target_reps_ = 12;
    int target_sets_ = 3;
};

}  // namespace fitness

#endif  // _FITNESS_EXERCISE_H_
