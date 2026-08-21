#include "../main/box_tracker.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using fall::boxCenterDistance;
using fall::boxIou;
using fall::greedyBoxAssignment;
using geometry::InferBox;

int main() {
    // Two people move slightly between frames.  The assignment follows the
    // boxes, not detector/list order, so a confidence re-sort cannot splice
    // either person's temporal history.
    const std::vector<InferBox> previous = {
        InferBox::fromCenter(0.25f, 0.50f, 0.20f, 0.45f, 0.8f),
        InferBox::fromCenter(0.72f, 0.48f, 0.18f, 0.42f, 0.9f),
    };
    const std::vector<InferBox> current = {
        InferBox::fromCenter(0.70f, 0.49f, 0.18f, 0.42f, 0.95f),
        InferBox::fromCenter(0.27f, 0.51f, 0.20f, 0.45f, 0.70f),
    };
    const auto assignment = greedyBoxAssignment(current, previous, 0.20f, 0.25f);
    assert(assignment.size() == 2);
    assert(assignment[0] == 1);
    assert(assignment[1] == 0);
    assert(boxIou(current[0], previous[1]) > 0.7f);
    assert(boxCenterDistance(current[1], previous[0]) < 0.05f);

    // A far-away detection cannot steal an existing track and is marked for a
    // fresh monotonically allocated track_id by MultiPersonTracker.
    const std::vector<InferBox> far = {
        InferBox::fromCenter(0.05f, 0.08f, 0.10f, 0.15f, 0.99f),
    };
    const auto new_assignment = greedyBoxAssignment(far, previous, 0.20f, 0.25f);
    assert(new_assignment.size() == 1 && new_assignment[0] == -1);

    std::cout << "multi_person_tracker_test: all scenarios passed\n";
    return 0;
}
