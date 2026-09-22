"""Basic regression tests for command_control.DroneCommandController.

Pure math, no hardware/MAVSDK/GStreamer required -- compute_command() takes a
bbox and frame dimensions and returns velocity floats plus a status string.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from command_control import DroneCommandController


def test_no_bbox_returns_zero_velocities_and_hover_string():
    controller = DroneCommandController()
    fwd, up, yaw, cmd_str = controller.compute_command(None, 1280, 720)
    assert (fwd, up, yaw) == (0.0, 0.0, 0.0)
    assert "HOVER" in cmd_str


def test_zero_frame_dimensions_returns_zero_velocities():
    controller = DroneCommandController()
    fwd, up, yaw, cmd_str = controller.compute_command((100, 100, 200, 200), 0, 0)
    assert (fwd, up, yaw) == (0.0, 0.0, 0.0)


def test_target_left_of_center_produces_nonzero_yaw():
    controller = DroneCommandController()
    frame_w, frame_h = 1280, 720
    # bbox centered well to the left of frame center
    bbox = (0, 300, 200, 500)
    fwd, up, yaw, cmd_str = controller.compute_command(bbox, frame_w, frame_h)
    assert yaw != 0.0


def test_target_at_frame_center_and_setpoint_area_yields_small_errors():
    controller = DroneCommandController()
    frame_w, frame_h = 1280, 720
    # Centered horizontally; vertically at the configured setpoint ratio;
    # sized to match the configured target area ratio -- should be close to
    # a "no correction needed" state on all three axes.
    import math
    setpoint_y = frame_h * controller.VERTICAL_SETPOINT_RATIO
    target_area = (frame_w * frame_h) * controller.TARGET_BBOX_AREA_RATIO
    side = math.sqrt(target_area)
    cx, cy = frame_w / 2, setpoint_y
    bbox = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    fwd, up, yaw, cmd_str = controller.compute_command(bbox, frame_w, frame_h)
    assert abs(yaw) < 0.01
    assert abs(up) < 0.01
    assert abs(fwd) < 0.01


def test_velocities_stay_within_configured_max_limits():
    controller = DroneCommandController()
    frame_w, frame_h = 1280, 720
    # A bbox far outside the frame's "normal" range to try to force clipping.
    bbox = (-5000, -5000, -4900, -4900)
    fwd, up, yaw, cmd_str = controller.compute_command(bbox, frame_w, frame_h)
    assert abs(yaw) <= controller.MAX_YAW_RATE
    assert abs(up) <= controller.MAX_VERTICAL_VEL
    assert abs(fwd) <= controller.MAX_FORWARD_VEL
