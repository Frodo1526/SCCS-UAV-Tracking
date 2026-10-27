from time import sleep

import cv2
import logging
from datetime import datetime
from collections import OrderedDict
from app_state import AppState

logging.basicConfig(filename='./app.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Colors in BGR format for OpenCV
COLOR_CONFIRMED = (0, 220, 0)  # Green bounding box for confirmed detections
COLOR_TENTATIVE = (0, 180, 180)  # Cyan bounding box for tentative detections
COLOR_TARGET = (0, 80, 255)  # Red bounding box for selected target
COLOR_HUD = (240, 240, 240)  # White HUD text


# Threaded UI Draw thread
def draw_overlay(tello, app_state: AppState, command_str=""):
    frame_reader = tello.get_frame_read()
    while not app_state.stop_event.is_set():
        # Bfrost 9.29
        with app_state.state_lock:
            state = app_state.drone_state
        with app_state.battery_lock:
            battery = app_state.battery
        with app_state.target_lock:
            target_id = app_state.target_id

        with app_state.tracker_lock:
            tracked = OrderedDict(app_state.tracked)

        frame = frame_reader.frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if target_id is None:
            target_id = 'none'

        if tracked is None:
            tracked = {0: {}}

        for object_id, data in tracked.items():
            bbox = data.get('bbox')
            label = data.get('label')
            conf = data.get('confidence', 0.0)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox

            is_target = (object_id == target_id)

            color = COLOR_TARGET if is_target else COLOR_CONFIRMED

            thickness = 3 if is_target else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            tag = f"ID:{object_id} {label} {conf:.2f}"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1 + 2, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, tag, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
            )

            if is_target:
                cx, cy = data['centroid']
                cv2.drawMarker(
                    frame, (cx, cy), COLOR_TARGET, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA
                )

        hud_lines = [
            f"STATE: {state}"
            f" TARGET: {target_id}"
            f" BATTERY: {battery}"
            " T = takeoff | L = land | Q = quit"
        ]

        manual_controls = [
            f"Manual Controls:",
            f"W/S: forward/backward",
            f"A/D: left/right",
            f"R/F: up/down",
        ]

        w = frame.shape[1]
        h = frame.shape[0]

        cv2.putText(
            frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], (w - 272, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 1, cv2.LINE_AA
        )

        for i, line in enumerate(hud_lines):
            cv2.putText(
                frame, line, (10, 40 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 1, cv2.LINE_AA
            )

        # Display current command values at the bottom
        if command_str:
            cv2.putText(
                frame, command_str, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 1, cv2.LINE_AA
            )

        if state == "MANUAL":
            for i, line in enumerate(manual_controls):
                cv2.putText(
                    frame, line, (10, 100 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUD, 1, cv2.LINE_AA
                )

        elif state == "TRACKING" and target_id != 'none':
            cv2.putText(
                frame, "| C Clear/Manual", (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUD, 1, cv2.LINE_AA
            )

        with app_state.frame_lock:
            app_state.frame = frame
        with app_state.frame_lock:
            app_state.display = frame.copy()

        sleep(.1)


def make_mouse_callback(app_state):
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        with app_state.tracker_lock:
            tracked = dict(app_state.tracked)
        with app_state.target_lock:
            for object_id, data in tracked.items():
                bbox = data.get('bbox')
                if bbox and bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    app_state.target_id = object_id
                    with app_state.state_lock:
                        app_state.drone_state = "TRACKING"
                    logging.info(f"Target acquired: ID {object_id} ({data.get('label')})")
                    return
            app_state.target_id = None
            with app_state.state_lock:
                app_state.drone_state = "MANUAL"
            logging.info("Target cleared.")
            return

    return on_mouse








