"""Basic regression tests for ctrl.compute_velocity_commands.

Pure math plus dict access -- no Tello hardware, no threads. app_state is
passed through but unused inside compute_velocity_commands, so None is fine.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from ctrl import compute_velocity_commands, DEADBAND_PX


FRAME_W, FRAME_H = 960, 720


def make_target(x1, y1, x2, y2):
    '''
    Builds a minimal target-data dict containing just a 'bbox', as
    expected by compute_velocity_commands(), for use in tests.

    Args:
        x1 (float): Left edge of the bounding box.
        y1 (float): Top edge of the bounding box.
        x2 (float): Right edge of the bounding box.
        y2 (float): Bottom edge of the bounding box.

    Returns:
        dict: {"bbox": (x1, y1, x2, y2)}.
    '''
    return {"bbox": (x1, y1, x2, y2)}

def test_no_bbox_returns_zero_and_hover_string():
    '''
    Verifies that compute_velocity_commands() returns all-zero commands
    and a status string containing "HOVER" when given an empty target
    dict (no 'bbox' key).

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via assert statements.
    '''
    fwd, vert, yaw, cmd_str = compute_velocity_commands({}, FRAME_W, FRAME_H, None)
    assert (fwd, vert, yaw) == (0, 0, 0)
    assert "HOVER" in cmd_str

def test_target_left_of_center_produces_nonzero_yaw():
    '''
    Verifies that a small target bounding box positioned well to the left
    of the frame center, and far from any frame edge, produces a non-zero
    yaw command.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via an assert statement.
    '''
    # Small bbox well to the left of center, far from any edge.
    target = make_target(100, 300, 160, 360)
    fwd, vert, yaw, cmd_str = compute_velocity_commands(target, FRAME_W, FRAME_H, None)
    assert yaw != 0

def test_target_dead_center_at_goal_area_yields_near_zero_commands():
    '''
    Verifies that a bounding box centered in the frame and sized to
    exactly match the configured goal area ratio produces zero forward,
    vertical, and yaw commands.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via assert statements.
    '''
    import math
    goal_area = FRAME_W * FRAME_H * 0.05
    side = math.sqrt(goal_area)
    cx, cy = FRAME_W / 2, FRAME_H / 2
    target = make_target(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    fwd, vert, yaw, cmd_str = compute_velocity_commands(target, FRAME_W, FRAME_H, None)
    assert fwd == 0
    assert vert == 0
    assert yaw == 0

@pytest.mark.xfail(strict=True)
def test_edge_proximity_suppresses_x_and_area_independently():
    '''
    Encodes the intended (but not-yet-implemented) behavior that a target
    near the left frame edge with an oversized bounding box should have
    both its x-error and its area-error suppressed by the deadband logic.
    Marked xfail(strict=True) because the current if/elif/elif
    implementation in compute_velocity_commands() only ever suppresses one
    of ex/ey/ea per call, so this test is expected to fail (and the suite
    fails if it unexpectedly starts passing without the marker being
    updated) until that chain is changed to independent if statements.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via an assert statement (expected,
            per the xfail marker).
    '''
    # Construct a target close to the LEFT edge (triggers ex suppression) AND
    # with a bbox larger than the goal area while near that same edge
    # (should ALSO trigger ea suppression, per the code's own comments).
    # A correct implementation suppresses both; the current if/elif chain
    # only suppresses ex.
    near_edge_x = DEADBAND_PX - 1  # inside the x-edge deadband
    cy = FRAME_H / 2
    # Oversized bbox (area >> goal area) so ea starts negative.
    target = make_target(0, cy - 200, near_edge_x + 200, cy + 200)

    fwd, vert, yaw, cmd_str = compute_velocity_commands(target, FRAME_W, FRAME_H, None)

    # If ea were correctly suppressed too, fwd should be 0 here. Today it
    # isn't, because the elif chain only ever suppresses ex for this input.
    assert fwd == 0
