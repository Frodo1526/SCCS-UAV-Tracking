from tello_poc.yolo26.detection.vehicle_types import (
    X1, Y1, X2, Y2,
    CONFIDENCE, CLASS_INDEX,
    VEHICLE_CLASSES
)
import time
import cv2
import logging
from collections import OrderedDict

MIN_BBOX_AREA = 2000

def yolo_inference(detector, frame, app_state, tracked_objects=None):
    # YOLO26 returns an (N, 6) NumPy array
    results = detector.detect(frame)

    detections = []
    h, w = frame.shape[:2]

    # Existing tracked objects can bypass edge filtering
    tracked_bboxes = [
        obj["bbox"]
        for obj in (tracked_objects or {}).values()
        if obj.get("bbox") is not None
    ]

    edge_exc = app_state.tracker.edge_margin

    for row in results:
        # Extract coordinates
        x1 = int(row[X1])
        y1 = int(row[Y1])
        x2 = int(row[X2])
        y2 = int(row[Y2])

        conf = float(row[CONFIDENCE])

        # Convert internal class index into a label string
        class_index = int(row[CLASS_INDEX])
        label = VEHICLE_CLASSES[class_index].display_name

        # Reject small bounding boxes
        bbox_area = (x2 - x1) * (y2 - y1)

        if bbox_area < MIN_BBOX_AREA:
            continue

        # Allow previously tracked objects near frame edges
        is_existing_track = any(
            x1 < bbox[2] and x2 > bbox[0]
            and y1 < bbox[3] and y2 > bbox[1]
            for bbox in tracked_bboxes
        )

        # Reject new detections too close to edges
        if not is_existing_track and (
            x1 < edge_exc or
            x2 > w - edge_exc or
            y1 < edge_exc or
            y2 > h - edge_exc
        ):
            continue

        # Calculate centroid
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Sample colors around centroid
        sample_size = int(((bbox_area / 2) ** 0.5) / 2)

        px0 = max(cx - sample_size, 0)
        py0 = max(cy - sample_size, 0)
        px1 = min(cx + sample_size, w)
        py1 = min(cy + sample_size, h)

        sample = frame[py0:py1, px0:px1]

        if sample.size == 0:
            continue

        color = sample.mean(axis=(0, 1)).tolist()

        # Return the format expected by your existing tracker
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "centroid": (cx, cy),
            "label": label,
            "confidence": conf,
            "color": color,
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
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
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