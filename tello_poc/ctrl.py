import numpy as np
import logging
import time

logging.basicConfig(filename = './app.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Control gains adapted from droneControl's DroneCommandController
YAW_GAIN = 1.5 # error_x (px) -> yaw_rate (deg/s)
UP_DOWN_GAIN = -1.5 # error_y (px) -> vertical_vel (cm/s)
FORWARD_GAIN = 1.4 # area_error (px^2) -> forward_vel (cm/s)
GOAL_AREA_RATIO = 0.05 # target bbox area as fraction of frame area

MAX_YAW_RATE = 100 # deg/s
MAX_VERTICAL_VEL = 100 # cm/s
MAX_FORWARD_VEL = 100 # cm/s
DEADBAND_PX = 20 # px, suppress commands when error is small

# State machine timeouts
HOVER_TIMEOUT = 4.0 # seconds in HOVERING before RETURNING
RETURN_HOLD_DURATION = 2.0 # seconds holding before back to MANUAL
CONTROL_RATE = 0.2 # seconds between control loop ticks (10 Hz)

FRAME_WIDTH = 960
FRAME_HEIGHT = 720

def compute_velocity_commands(target_data, frame_width, frame_height, app_state):
    '''
    Computes forward/vertical/yaw RC velocity commands (in the units and
    scale expected by djitellopy's send_rc_control) that steer the Tello
    to center on and maintain a target size for a tracked object's
    bounding box.

    If target_data has no 'bbox', returns a zero-command hover result.
    Otherwise computes the target's centroid and area, the pixel errors
    between the centroid and the frame center (ex, ey) and between the
    bbox area and a goal area (ea, GOAL_AREA_RATIO of frame area), then
    applies a deadband/edge-proximity suppression: if the target's
    centroid is within DEADBAND_PX of the nearest vertical edge, ex is
    zeroed; otherwise if it's within DEADBAND_PX of the nearest horizontal
    edge, ey is zeroed; otherwise if the target is larger than the goal
    area and within 2*DEADBAND_PX of either edge, ea is zeroed. (Only one
    of these three branches can apply per call, since they are chained
    with elif.) Additionally, ex and ey are independently zeroed outright
    if their magnitude is already below DEADBAND_PX. Errors are then
    normalized and scaled by fixed gains, clipped to MAX_YAW_RATE /
    MAX_VERTICAL_VEL / MAX_FORWARD_VEL, and rounded to integers.

    Args:
        target_data (dict): Tracked target's detection info; must contain
            'bbox' as (x1, y1, x2, y2) for a non-hover result.
        frame_width (int): Width of the video frame in pixels.
        frame_height (int): Height of the video frame in pixels.
        app_state: Accepted for a consistent call signature but not read
            or used anywhere in this function's body.

    Returns:
        tuple[int, int, int, str]: (fwd, vert, yaw, cmd_str) -- forward,
            vertical, and yaw RC command values as rounded integers
            (cm/s and deg/s respectively), and a human-readable summary
            string. When target_data has no 'bbox', this is
            (0, 0, 0, a "HOVER (no bbox)!" string) instead.
    '''
    bbox = target_data.get('bbox')
    if bbox is None:
        return 0, 0, 0, "HOVER (no bbox)!"
    
    x1, y1, x2, y2 = bbox # left, top, right, bottom
    
    cx = (x1 + x2) / 2 # target centroid x coordinate
    cy = (y1 + y2) / 2 # target centroid y coordinate
    area = (x2 - x1) * (y2 - y1) # target bounding box area
    
    frame_cx = frame_width / 2 # frame centroid x coordinate
    frame_cy = frame_height / 2 # frame centroid y coordinate
    
    goal_area = frame_width * frame_height * GOAL_AREA_RATIO # Desired area of target as percentage of frame
    
    ex = cx - frame_cx # Pixel-distance between frame centroid and target centroid in x direction
    ey = cy - frame_cy # Pixel-distance between frame centroid and target centroid in y direction
    
    ea = goal_area - area # Pixel-area difference between goal area and actual target bbox area
    if -0.1 < ea < 0.1:
        logging.info("Close to goal area, FWD stalled")
    
    edge_proximity_x = min(cx, frame_width - cx) # Distance from target centroid to closest vertical edge
    edge_proximity_y = min(cy, frame_height - cy) # Distance from target centroid to closest horizontal edge
    
    if edge_proximity_x < DEADBAND_PX: # If target is close to vertical edge, suppress x error to avoid erratic yaw commands
        ex = 0
    elif edge_proximity_y < DEADBAND_PX: # If target is close to horizontal edge, suppress y error to avoid erratic vertical commands
        ey = 0
    elif ea < 0 and (edge_proximity_x < DEADBAND_PX * 2 or edge_proximity_y < DEADBAND_PX * 2): # If target is larger than goal area and close to any edge, suppress area error to avoid aggressive forward commands
        ea = 0
    
    if abs(ex) < DEADBAND_PX: ex = 0 # Issue a 0 command in this direction if error is small enough, reducing erratic behavior
    if abs(ey) < DEADBAND_PX: ey = 0 # Issue a 0 command in this direction if error is small enough, reducing erratic behavior
    
    ex_norm = ex / (frame_width / 2) # Normalized x error as a fraction of frame_width / 2, yielding a value in [-1, 1]
    ey_norm = ey / (frame_height / 2) # Normalized y error as a fraction of frame_height / 2, yielding a value in [-1, 1]
    ea_norm = ea / (frame_width * frame_height * GOAL_AREA_RATIO)
    
    yaw  = float(np.clip(MAX_YAW_RATE * YAW_GAIN * ex_norm, -MAX_YAW_RATE, MAX_YAW_RATE)) # Yaw rate command in deg/s, clipped to max yaw rate
    vert = float(np.clip(MAX_VERTICAL_VEL * UP_DOWN_GAIN * ey_norm, -MAX_VERTICAL_VEL, MAX_VERTICAL_VEL)) # Vertical velocity command in cm/s, clipped to max vertical velocity
    fwd  = float(np.clip(MAX_FORWARD_VEL * FORWARD_GAIN * ea_norm, -MAX_FORWARD_VEL, MAX_FORWARD_VEL)) # Forward velocity command in cm/s, clipped to max forward velocity
    
    cmd_str = f"FWD={fwd:.0f}cm/s | VERT={vert:.0f}cm/s | YAW={yaw:.0f}°/s"
    return round(fwd), round(vert), round(yaw), cmd_str # Return rounded integer commands for tello.send_rc_control, along with a string for logging

def run_manual_control(tello, app_state):
    '''
    Sends one RC command to the Tello based on the set of currently
    pressed manual-control keys (W/S for forward/backward, A/D for yaw
    left/right, R/F for up/down), each contributing a fixed +/-60
    magnitude; combinations are summed simply by which flags were set
    (later checks for the same axis overwrite earlier ones, so opposite
    keys pressed simultaneously resolve to whichever was checked last: S
    overrides W, D overrides A, F overrides R). Returns immediately
    without sending anything if app_state.stop_event or
    app_state.control_stop is set. If the drone is airborne, updates
    app_state.current_command_str and calls
    tello.send_rc_control(0, fwd, vert, yaw); if not airborne, the
    command string/state is left as the caller set it and no RC command
    is sent.

    Args:
        tello (djitellopy.Tello): Tello SDK client used to send the RC
            control command.
        app_state (AppState): Shared application state providing
            pressed_keys (and its lock), airborne, command_lock and
            current_command_str, and the stop events.

    Returns:
        None: This function does not return a value.
    '''
    """Send RC commands based on currently pressed keys."""
    logging.info("State: MANUAL (looping on key presses)")
    
    if app_state.stop_event.is_set() or app_state.control_stop.is_set():
        return
    
    # Initialize all movement commands to 0
    fwd = 0
    vert = 0
    yaw = 0
    
    with app_state.keys_lock:
        pressed_keys = app_state.pressed_keys.copy()
    
    # Check for movement commands (can be combined)
    if ord('W') in pressed_keys or ord('w') in pressed_keys:
        fwd = 60
    if ord('S') in pressed_keys or ord('s') in pressed_keys:
        fwd = -60
    if ord('A') in pressed_keys or ord('a') in pressed_keys:
        yaw = -60
    if ord('D') in pressed_keys or ord('d') in pressed_keys:
        yaw = 60
    if ord('R') in pressed_keys or ord('r') in pressed_keys:
        vert = 60
    if ord('F') in pressed_keys or ord('f') in pressed_keys:
        vert = -60
    
    if app_state.airborne:
        cmd_str = f"FWD={fwd}cm/s | VERT={vert}cm/s | YAW={yaw}°/s"
        with app_state.command_lock:
            app_state.current_command_str = cmd_str
        tello.send_rc_control(0, fwd, vert, yaw)

def run_control_loop(tello, app_state):
    '''
    Runs the Tello subsystem's main control loop at CONTROL_RATE-second
    intervals until app_state.stop_event or app_state.control_stop is
    set: reads the current drone_state, target_id, the target's tracked
    data, and the current frame under their respective locks, then
    dispatches based on drone_state:

      - "MANUAL": delegates to run_manual_control() and clears the
        displayed command string.
      - "TRACKING": if target_id is None, sends a zero RC command and
        transitions to "MANUAL". If the target's tracked data is missing
        (target lost), transitions to "HOVERING" (recording the loss
        time) and, if airborne, sends a zero RC command. Otherwise,
        computes RC commands via compute_velocity_commands(), updates the
        displayed command string, and sends them via
        tello.send_rc_control() if airborne.
      - "HOVERING": clears the displayed command string. If the target's
        tracked data has reappeared, transitions back to "TRACKING".
        Otherwise, if HOVER_TIMEOUT seconds have elapsed since the target
        was lost, transitions to "RETURNING" (recording the hold start
        time) and, if airborne, sends a zero RC command; if not yet timed
        out, sends a zero RC command (if airborne) and logs the remaining
        time.
      - "RETURNING": once RETURN_HOLD_DURATION seconds have elapsed since
        the hold started, transitions to "MANUAL" and clears target_id.

    Exceptions raised inside a single iteration are caught, logged (unless
    the stop event is set), and the loop continues after the usual sleep.
    On exit, sends one final zero RC command (best-effort, any exception
    is swallowed).

    Args:
        tello (djitellopy.Tello): Tello SDK client used to send RC control
            commands.
        app_state (AppState): Shared application state providing
            drone_state, target_id, tracked, frame (each with its lock),
            hover_lost_time, hold_start_time, current_command_str (and its
            lock), airborne, and the stop events.

    Returns:
        None: This function does not return a value; it runs until one of
            the stop events is set.
    '''
    logging.info("Control loop started.")
    
    control_runtime_ds = 0 # control loop runtime in deciseconds

    while not app_state.stop_event.is_set() and not app_state.control_stop.is_set():
        try:
            with app_state.state_lock:
                state = app_state.drone_state
            with app_state.target_lock:
                target_id = app_state.target_id
            with app_state.tracker_lock:
                target_data = app_state.tracked.get(target_id) if target_id else None
            with app_state.frame_lock:
                frame = app_state.frame

            fw = FRAME_WIDTH
            fh = FRAME_HEIGHT

            if state == "MANUAL":
                logging.info("State: MANUAL")
                run_manual_control(tello, app_state)
                with app_state.command_lock:
                    app_state.current_command_str = ""
                    
            elif state == "TRACKING":
                if target_id is None:
                    with app_state.state_lock:
                        tello.send_rc_control(0, 0, 0, 0)
                        app_state.drone_state = "MANUAL"
                    logging.info("[TRACKING] -> [MANUAL]: target cleared.")

                elif target_data is None:
                    with app_state.state_lock:
                        app_state.drone_state = "HOVERING"
                        app_state.hover_lost_time = time.time()
                    logging.info("[TRACKING] → [HOVERING]: target lost.")
                    if app_state.airborne:
                        tello.send_rc_control(0, 0, 0, 0)
                else:
                    fwd, vert, yaw, cmd_str = compute_velocity_commands(target_data, fw, fh, app_state)
                    # Update display with current command values
                    with app_state.command_lock:
                        app_state.current_command_str = cmd_str
                    if app_state.airborne:
                        # send_rc_control(left_right, fwd_back, up_down, yaw)
                        tello.send_rc_control(0, fwd, vert, yaw)

            elif state == "HOVERING":
                elapsed   = time.time() - app_state.hover_lost_time
                remaining = HOVER_TIMEOUT - elapsed
                
                with app_state.command_lock:
                    app_state.current_command_str = ""

                if target_data is not None:
                    with app_state.state_lock:
                        app_state.drone_state = "TRACKING"
                    app_state.hover_lost_time = None
                    logging.info("[HOVERING] -> [TRACKING]: target reacquired.")

                elif elapsed >= HOVER_TIMEOUT:
                    with app_state.state_lock:
                        app_state.drone_state = "RETURNING"
                        app_state.hold_start_time = time.time()
                    logging.info("[HOVERING] -> [RETURNING]: timeout.")
                    if app_state.airborne:
                        tello.send_rc_control(0, 0, 0, 0)
                else:
                    logging.debug(f"[HOVERING] {remaining:.1f}s remaining.")
                    if app_state.airborne:
                        tello.send_rc_control(0, 0, 0, 0)

            elif state == "RETURNING":
                elapsed = time.time() - app_state.hold_start_time
                if elapsed >= RETURN_HOLD_DURATION:
                    with app_state.state_lock:
                        app_state.drone_state = "MANUAL"
                    with app_state.target_lock:
                        app_state.target_id = None
                    app_state.hold_start_time = None
                    logging.info("[RETURNING] -> [MANUAL].")
                    
            control_runtime_ds += 1 # Add 1 to running time

        except Exception as e:
            if not app_state.stop_event.is_set():
                logging.error(f"Control loop error: {e}")

        time.sleep(CONTROL_RATE)
    
    try:
        tello.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass

    logging.info("Control loop stopped.")