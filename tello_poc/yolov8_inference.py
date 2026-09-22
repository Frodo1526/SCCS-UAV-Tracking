import time
import cv2
import logging
from collections import OrderedDict

YOLO_CONFIDENCE = 0.35
YOLO_CLASSES = [2, 5, 7, 0]
MIN_BBOX_AREA = 2000

def yolo_inference(model, frame, app_state, tracked_objects=None):
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
    