import cv2
import logging
from datetime import datetime

logging.basicConfig(filename = './app.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Colors in BGR format for OpenCV
COLOR_CONFIRMED = (0, 220, 0) # Green bounding box for confirmed detections
COLOR_TENTATIVE = (0, 180, 180) # Cyan bounding box for tentative detections
COLOR_TARGET = (0, 80, 255) # Red bounding box for selected target
COLOR_HUD = (240, 240, 240) # White HUD text

def draw_overlay(frame, tracked, target_id, state, battery, command_str=""):
    '''
    Draws the Tello POC's on-frame HUD: a bounding box, ID/label/confidence
    tag, and (for the current target) a crosshair marker for every tracked
    object; a timestamp; a status line (state, target, battery, key
    hints); the current command string if non-empty; and, depending on
    `state`, either a manual-controls legend (MANUAL) or a "clear target"
    hint (TRACKING with a real target selected).

    If target_id is None, it is displayed as the string 'none' for the
    status line. If tracked is None, it is replaced with a single
    placeholder entry ({0: {}}) whose bbox is None, which is then skipped
    by the per-object drawing loop (so no boxes are drawn in that case).
    Entries in tracked whose 'bbox' is None are skipped.

    Args:
        frame (numpy.ndarray): The BGR video frame (OpenCV format) to draw
            onto; mutated in place and also returned.
        tracked (dict | None): Mapping of object ID to tracked-object info
            dict (expected keys: 'bbox', 'label', 'confidence', and, for
            the target, 'centroid'). None is treated as an empty
            placeholder.
        target_id: ID of the currently selected target (or None/'none' for
            no target), used to choose highlight color and draw the
            crosshair.
        state (str): Current drone state machine value (e.g. "MANUAL",
            "TRACKING"), shown in the HUD and used to decide which
            control-hint text to show.
        battery (int): Current battery percentage, shown in the HUD.
        command_str (str): Current RC command summary string to display at
            the bottom of the frame; nothing is drawn for this if empty.
            Defaults to "".

    Returns:
        numpy.ndarray: The same frame object passed in, with all overlay
            drawing applied in place.
    '''
    if target_id is None:
        target_id = 'none'
        
    if tracked is None:
        tracked = {0 : {}}
    
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
        
    return frame

def make_mouse_callback(app_state):
    '''
    Builds and returns an OpenCV mouse-callback function (closing over
    app_state) suitable for cv2.setMouseCallback(), which selects a
    tracked object as the target on left-button click, or clears the
    target if the click misses every tracked bounding box.

    Args:
        app_state (AppState): Shared application state the returned
            callback will read/update (tracked, tracker_lock, target_id,
            target_lock, drone_state, state_lock).

    Returns:
        Callable[[int, int, int, int, object], None]: A callback matching
            OpenCV's mouse-callback signature (event, x, y, flags, param),
            for use with cv2.setMouseCallback().
    '''
    def on_mouse(event, x, y, flags, param):
        '''
        OpenCV mouse-callback handler (returned by make_mouse_callback).
        Ignores every event except a left-button-down click. On a qualifying
        click, takes a snapshot of app_state.tracked and, in iteration order,
        selects the first object whose bbox contains (x, y) as the new target
        -- setting app_state.target_id to that object's ID and
        app_state.drone_state to "TRACKING". If no object's bbox contains the
        click, clears app_state.target_id to None and sets
        app_state.drone_state to "MANUAL" instead.

        Args:
            event (int): OpenCV mouse event type constant; only
                cv2.EVENT_LBUTTONDOWN is handled.
            x (int): X-coordinate of the mouse event, in frame pixel space.
            y (int): Y-coordinate of the mouse event, in frame pixel space.
            flags (int): OpenCV event flags; unused.
            param: OpenCV callback user-data parameter; unused.

        Returns:
            None: This function does not return a value; it updates
                app_state.target_id and app_state.drone_state as side
                effects.
        '''
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