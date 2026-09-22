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
    return {"bbox": (x1, y1, x2, y2)}


def test_no_bbox_returns_zero_and_hover_string():
    fwd, vert, yaw, cmd_str = compute_velocity_commands({}, FRAME_W, FRAME_H, None)
    assert (fwd, vert, yaw) == (0, 0, 0)
    assert "HOVER" in cmd_str


def test_target_left_of_center_produces_nonzero_yaw():
    # Small bbox well to the left of center, far from any edge.
    target = make_target(100, 300, 160, 360)
    fwd, vert, yaw, cmd_str = compute_velocity_commands(target, FRAME_W, FRAME_H, None)
    assert yaw != 0


def test_target_dead_center_at_goal_area_yields_near_zero_commands():
    import math
    goal_area = FRAME_W * FRAME_H * 0.05
    side = math.sqrt(goal_area)
    cx, cy = FRAME_W / 2, FRAME_H / 2
    target = make_target(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    fwd, vert, yaw, cmd_str = compute_velocity_commands(target, FRAME_W, FRAME_H, None)
    assert fwd == 0
    assert vert == 0
    assert yaw == 0


@pytest.mark.xfail(
    reason=(
        "Known bug from the original code review: the edge-proximity deadband "
        "suppression in compute_velocity_commands uses an if/elif/elif chain, "
        "so only ONE of ex/ey/ea ever gets suppressed per call even when a "
        "target triggers more than one condition at once. This test encodes "
        "the INTENDED behavior (independent suppression per axis) and will "
        "start passing once the if/elif/elif is changed to three independent "
        "if statements -- see the code review's Correctness section, #2.2."
    ),
    strict=True,
)
def test_edge_proximity_suppresses_x_and_area_independently():
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
