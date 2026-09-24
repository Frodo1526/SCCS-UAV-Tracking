import time
import cv2
import logging
from collections import OrderedDict

YOLO_CONFIDENCE = 0.35
YOLO_CLASSES = [2, 5, 7, 0]
MIN_BBOX_AREA = 2000

def yolo_inference(model, frame, app_state, tracked_objects=None):
    '''
    Runs one YOLOv8 inference pass over a frame and converts the results
    into the detection-info dict format expected by CentroidTracker.

    Detections are filtered by the model call itself to YOLO_CONFIDENCE
    and the fixed YOLO_CLASSES class-index allow-list. Additionally,
    detections smaller than MIN_BBOX_AREA are discarded, and detections
    that are near the frame edge (within app_state.tracker.edge_margin
    pixels of any side) are discarded UNLESS they overlap the bounding box
    of a currently tracked object -- this lets existing tracks continue to
    be reported as they move toward the frame boundary, while still
    filtering out spurious new near-edge detections.

    For each kept detection, a square patch centered on its centroid
    (sized to half the bbox's area, then halved again for the sample
    region) is averaged to produce a representative color, clipped to stay
    within the frame.

    Args:
        model: A loaded Ultralytics YOLO model, called as
            model(frame, verbose=False, conf=YOLO_CONFIDENCE, classes=YOLO_CLASSES).
        frame (numpy.ndarray): BGR video frame (OpenCV format) to run
            inference on.
        app_state (AppState): Shared application state; only
            app_state.tracker.edge_margin is read.
        tracked_objects (dict, optional): Mapping of ID to currently
            tracked object info dicts (each with a 'bbox'), used to bypass
            edge filtering for detections overlapping an existing track.
            Defaults to None, which disables the edge-filtering bypass
            entirely (no detection is treated as an existing track).

    Returns:
        list[dict]: One dict per kept detection, each with 'bbox'
            (x1, y1, x2, y2), 'centroid' (cx, cy), 'label' (str),
            'confidence' (float), and 'color' (list[float], BGR mean).
    '''
    results = model(frame, verbose=False, conf=YOLO_CONFIDENCE, classes=YOLO_CLASSES) # list of Results objects
    detections = []
    h, w = frame.shape[:2]
    
    # Get bounding boxes of currently tracked objects for edge detection bypass
    tracked_bboxes = []
    if tracked_objects:
        tracked_bboxes = [obj.get('bbox') for obj in tracked_objects.values() if obj.get('bbox')]
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) # Pixel coordinates from bounding box coordinate as a list
        conf = float(box.conf[0]) # Confidence score as float
        label = results[0].names[int(box.cls[0])] # Class label as string
        
        edge_exc = app_state.tracker.edge_margin # Exclude boxes too close to edges to improve tracking stability
        
        if (x2 - x1) * (y2 - y1) < MIN_BBOX_AREA:
            continue
        
        # Check if this detection overlaps with any currently tracked object
        # If so, allow it even near edges (to prevent losing tracked targets at frame boundaries)
        is_existing_track = any(
            x1 < bbox[2] and x2 > bbox[0] and y1 < bbox[3] and y2 > bbox[1]
            for bbox in tracked_bboxes
        )
        
        # Only apply edge filtering to NEW detections, not existing tracks
        if not is_existing_track and (x1 < edge_exc or x2 > w - edge_exc or y1 < edge_exc or y2 > h - edge_exc):
            continue
        
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        # 1/2 Bbox area color sample around centroid, clipped to frame boundaries
        bbox_area = (x2 - x1) * (y2 - y1)
        sample_size = int(((bbox_area / 2) ** 0.5) / 2) # Sample size is half the side length of a square with area equal to half the bbox area
        px0 = max(cx - sample_size, 0)
        py0 = max(cy - sample_size, 0)
        px1 = min(cx + sample_size, w)
        py1 = min(cy + sample_size, h)
        color = frame[py0:py1, px0:px1].mean(axis=(0, 1)).tolist()
        
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'centroid': (cx, cy),
            'label': label,
            'confidence': conf,
            'color': color,
        })

    return detections

def run_inference_loop(model, frame_read, app_state, FRAME_WIDTH, FRAME_HEIGHT):
    '''
    Runs the Tello POC's continuous inference loop until
    app_state.stop_event is set: reads the latest raw frame from the
    Tello's frame_read (skipping the iteration with a short sleep if no
    frame is available yet), converts it from RGB to BGR and resizes it to
    (FRAME_WIDTH, FRAME_HEIGHT), runs yolo_inference() against a snapshot
    of the currently tracked objects, then updates the shared
    CentroidTracker -- via update_target() if a target is currently
    selected, or update_all_detections() otherwise. If the tracker update
    yields no tracked objects at all, and the target selected before this
    update is still the current target, the target is cleared (handles
    the case where the previously tracked target has been dropped by the
    tracker). Finally publishes the new tracked-object snapshot and the
    processed frame to app_state under their respective locks, and records
    app_state.last_frame_time.

    Exceptions raised during an iteration are caught, logged (with
    traceback), and followed by a short sleep before continuing.

    Args:
        model: A loaded Ultralytics YOLO model, passed through to
            yolo_inference().
        frame_read: The Tello SDK's frame-reader object; its `.frame`
            attribute is polled for the latest raw camera frame.
        app_state (AppState): Shared application state providing
            stop_event, tracker_lock/tracked, target_lock/target_id,
            frame_lock/frame, and last_frame_time.
        FRAME_WIDTH (int): Width to resize each frame to before inference
            and tracking.
        FRAME_HEIGHT (int): Height to resize each frame to before
            inference and tracking.

    Returns:
        None: This function does not return a value; it runs until
            app_state.stop_event is set.
    '''
    while not app_state.stop_event.is_set():
        try:
            raw = frame_read.frame
            if raw is None:
                time.sleep(0.01)
                continue
            
            app_state.last_frame_time = time.time()
            
            # Tello streams in RGB, convert to BGR for OpenCV
            raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(raw, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # Get currently tracked objects to allow edge detections for existing tracks
            with app_state.tracker_lock:
                tracked_snapshot = OrderedDict(app_state.tracked)
            
            detections = yolo_inference(model, frame, app_state, tracked_snapshot)
            
            # Get current target outside the tracker lock
            with app_state.target_lock:
                current_target = app_state.target_id
            
            # Run heavy inference outside the lock to avoid blocking display thread
            if current_target:
                tracked_objects = app_state.tracker.update_target(
                    detections, current_target, FRAME_WIDTH, FRAME_HEIGHT
                )
            else:
                tracked_objects = app_state.tracker.update_all_detections(
                    detections, FRAME_WIDTH, FRAME_HEIGHT
                )
            
            if not tracked_objects:
                with app_state.target_lock:
                    if app_state.target_id == current_target:
                        app_state.target_id = None
            
            # Only hold lock briefly to write results
            with app_state.tracker_lock:
                app_state.tracked = OrderedDict(tracked_objects)
            
            with app_state.frame_lock:
                app_state.frame = frame
                
        except Exception as e:
            logging.error(f"Inference loop error (continuing): {e}", exc_info=True)
            time.sleep(0.1)