import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import logging

logging.basicConfig(filename = './app.log', level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

# Clone of droneControl project's centroid_tracker.py
class CentroidTracker:
    def __init__(
        self,
        max_disappeared=20,
        tight_distance_ratio=0.05,
        max_distance_ratio=0.10,
        hit_streak_required=3,
        velocity_decay=0.5,
        edge_margin=20,
        next_id_counter=1,
        max_color_distance=15.0,
        iou_suppresion_thresh=0.6
    ):
        # Confirmed tracks, keyed by object ID with detection info dicts
        self.tracked_objects = OrderedDict() # Mapping object ids to detection info dicts
        self.disappeared_frames = OrderedDict() # Mapping object ids to missed frame counts
        self.velocities = OrderedDict() # Mapping object ids to velocity tuples
        self.colors = OrderedDict() # Mapping confirmed tracks to color data from frame

        # Tentative tracks, keyed by tentative id with detection info dicts and hit counts
        self._tentative_objects = OrderedDict() # Mapping tent ids to detection info dicts
        self._tentative_hits = OrderedDict() # Mapping tent ids to consecutive hit counts
        self._tentative_id_seq = 0 # Negative ID sequence for tentative tracks to avoid collision with confirmed IDs


        self.max_distance_ratio  = max_distance_ratio # Max distance from last centroid to consider a match, ratio of frame width
        self.tight_distance_ratio = tight_distance_ratio
        self.max_color_distance = max_color_distance
        self.hit_streak_required = hit_streak_required # Consecutive matches required to append tentative match to confirmed tracks
        self.max_disappeared_frames = max_disappeared  # max_disappeared: frames a confirmed track can go unmatched before deregistration
        self.iou_suppresion_thresh = iou_suppresion_thresh
        self.velocity_decay = velocity_decay # Per frame decay factor for velocity to prevent drift during occlusion, 0.5 means velocity halves each unmatched frame
        self.edge_margin = edge_margin # Margin in pixels where velocity influence is reduced to cleanly stop tracking
        self.next_id_counter = next_id_counter # Incrementing id counter to give each object a unique ID

    def update_all_detections(self, current_detections_info, frame_width, frame_height):
        """
        Main update step. Call once per frame with all filtered detections.

        Matching runs in two passes:
          Pass 1 — match detections against confirmed tracks (priority)
          Pass 2 — match remaining detections against tentative tracks
          Remainder — create new tentative entries

        Returns only confirmed tracked_objects (same dict as before)
        
        :param current_detections_info: list of detection info dicts with at least 'centroid' key
        :param frame_width: width of the video frame (for distance normalization)
        :param frame_height: height of the video frame (for distance normalization)
        :return: dict of confirmed tracked objects with their detection info
        """
        max_distance = frame_width * self.max_distance_ratio
        tight_distance = frame_width * self.tight_distance_ratio
        
        if not current_detections_info: # If there are no detections, age all confirmed and tentative tracks and return
            self._age_confirmed(set())
            self._age_tentative(set())
            return self.tracked_objects

        input_centroids = np.array([d['centroid'] for d in current_detections_info]) # Current frame centroid vectors
        
        # Validate centroid values for NaN/Inf
        if not np.all(np.isfinite(input_centroids)):
            logging.warning("Invalid detection centroids detected (NaN or Inf values)")
            self._age_confirmed(set())
            self._age_tentative(set())
            return self.tracked_objects

        # First pass, match against confirmed tracks using Hungarian algorithm with weighted scoring
        unmatched_detection_cols = set(range(len(current_detections_info)))
        matched_detection_cols = set()
        matched_confirmed_rows = set()

        if self.tracked_objects:
            confirmed_ids = list(self.tracked_objects.keys()) 
            predicted = self._build_predicted_centroids(confirmed_ids, frame_width, frame_height)
            prev_centroids = np.array(predicted) # Predicted centroid vectors for confirmed tracks

            # Compute distance matrix between predicted centroids and current detections
            distance = cdist(prev_centroids, input_centroids)
            
            # Build cost matrix using weighted scoring function
            # Each cell (i,j) = cost of assigning confirmed track i to detection j
            cost_matrix = np.full((len(confirmed_ids), len(current_detections_info)), 2.0)
            
            for row, object_id in enumerate(confirmed_ids):
                for col, detection in enumerate(current_detections_info):
                    score = self._compute_match_score(
                        object_id, detection, distance[row, col],
                        max_distance, tight_distance, frame_width, frame_height
                    )
                    cost_matrix[row, col] = score
            
            # Apply Hungarian algorithm for optimal assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Process optimal assignments: only accept matches with valid score < 1.0
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 1.0:  # Only accept reasonable matches
                    object_id = confirmed_ids[row]
                    logging.debug(f"Matched confirmed track {object_id} to detection {col} with score {cost_matrix[row, col]:.3f}")
                    self._update_confirmed_track(object_id, current_detections_info[col])
                    matched_confirmed_rows.add(row)
                    matched_detection_cols.add(col)
                else:
                    logging.debug(f"Rejected assignment (row={row}, col={col}) with score {cost_matrix[row, col]:.3f}")

            self._age_confirmed(matched_confirmed_rows)
            unmatched_detection_cols -= matched_detection_cols

        else:
            self._age_confirmed(set())
            
        filtered_unmatched = set()
        for col in unmatched_detection_cols: # Avoid registering unmatched detections that are duplicates of confirmed tracks by checking for overlap
            det_bbox = current_detections_info[col]['bbox']
            overlaps_confirmed = any(
                self._compute_iou(det_bbox, self.tracked_objects[object_id]['bbox']) > self.iou_suppresion_thresh
                for object_id in self.tracked_objects
            )
            if not overlaps_confirmed:
                filtered_unmatched.add(col)
        
        unmatched_detection_cols = filtered_unmatched
            
        # Second pass, match remaining detections against tentative tracks using scoring
        still_unmatched_cols = filtered_unmatched
        
        if self._tentative_objects and unmatched_detection_cols: # Try to match if tentative tracks exist and unmatched detections remain
            tent_ids = list(self._tentative_objects.keys())
            tent_centroids = np.array([
                self._tentative_objects[tid]['centroid'] for tid in tent_ids
            ])
            unmatched_col_list = sorted(unmatched_detection_cols)
            unmatched_input = np.array([
                current_detections_info[c]['centroid'] for c in unmatched_col_list
            ])

            tent_dist = cdist(tent_centroids, unmatched_input)
            
            # Build cost matrix for tentative tracks with simplified scoring (no velocity)
            tent_cost_matrix = np.full((len(tent_ids), len(unmatched_col_list)), 2.0)
            
            for row, tent_id in enumerate(tent_ids):
                for col_idx, orig_col in enumerate(unmatched_col_list):
                    dist = tent_dist[row, col_idx]
                    
                    # Hard constraint: spatial distance
                    if dist > max_distance:
                        tent_cost_matrix[row, col_idx] = 2.0
                        continue
                    
                    # Hard constraint: label must match
                    tent_label = self._tentative_objects[tent_id].get('label')
                    detection_label = current_detections_info[orig_col].get('label')
                    if tent_label and detection_label and tent_label != detection_label:
                        tent_cost_matrix[row, col_idx] = 2.0
                        continue
                    
                    # Soft score: simplified (just spatial + color for tentative)
                    spatial_score = min(1.0, dist / max_distance) * 0.6
                    
                    color_distance = self._calculate_color_distance_tentative(tent_id, current_detections_info[orig_col])
                    color_deficit = np.clip(color_distance / self.max_color_distance, 0.0, 1.0)
                    color_score = color_deficit * 0.4
                    
                    tent_cost_matrix[row, col_idx] = spatial_score + color_score
            
            # Apply Hungarian algorithm for tentative matching
            tent_row_ind, tent_col_ind = linear_sum_assignment(tent_cost_matrix)
            
            matched_tent_rows = set()
            matched_tent_cols = set()
            
            for row, col in zip(tent_row_ind, tent_col_ind):
                if tent_cost_matrix[row, col] < 1.0:  # Accept reasonable matches
                    tent_id = tent_ids[row]
                    original_col = unmatched_col_list[col]
                    
                    self._tentative_objects[tent_id] = current_detections_info[original_col]
                    self._tentative_hits[tent_id] += 1
                    
                    logging.debug(
                        f"Tentative {tent_id} hit streak: "
                        f"{self._tentative_hits[tent_id]}/{self.hit_streak_required} (score: {tent_cost_matrix[row, col]:.3f})"
                    )
                    
                    if self._tentative_hits[tent_id] >= self.hit_streak_required:
                        self._promote_tentative(tent_id)
                    
                    matched_tent_rows.add(row)
                    matched_tent_cols.add(col)
                    still_unmatched_cols.discard(original_col)
            
            self._age_tentative(matched_tent_rows)

        elif self._tentative_objects:
            # Tentative tracks exist but no detections left to match — discard all
            self._age_tentative(set())

        # Any detections still unmatched after both passes become new tentative tracks
        for col in still_unmatched_cols:
            self._register_tentative(current_detections_info[col])

        return self.tracked_objects
    
    def update_target(self, current_detections_info, target_id, frame_width, frame_height):
        """
        Update target detection only. Call once per frame while tracking.
        Called when tracking for lighter computation overhead rather than determining matches
        of every detection within the frame.
        
        """
        
        if not current_detections_info:
            self.disappeared_frames[target_id] += 1
            if self.disappeared_frames[target_id] > self.max_disappeared_frames:
                self._deregister(target_id)
            else:
                vx, vy = self.velocities.get(target_id, (0,0))
                self.velocities[target_id] = (vx * self.velocity_decay, vy * self.velocity_decay)
            return {}
            
        if target_id not in self.tracked_objects.keys():
            logging.info(f"Target {target_id} not found")
            return {}
        
        max_distance = frame_width * self.max_distance_ratio
        tight_distance = frame_width * self.tight_distance_ratio
        
        #Single-object predicted centroid computation
        input_centroids = np.array([d['centroid'] for d in current_detections_info])
        cx, cy = self.tracked_objects[target_id]['centroid']
        vx, vy = self.velocities.get(target_id, (0,0))
        
        edge_proximity  = min(cx, frame_width - cx, cy, frame_height - cy) # Distance to nearest edge of the frame
        velocity_weight = float(np.clip(edge_proximity / self.edge_margin, 0.0, 1.0))
        
        px = int(np.clip(cx + vx * velocity_weight, 0, frame_width - 1))  
        py = int(np.clip(cy + vy * velocity_weight, 0, frame_height - 1))
        
        predicted_centroid = np.array([px, py])
        predicted_centroid_2d = predicted_centroid.reshape(1, -1)
        
        distance = cdist(predicted_centroid_2d, input_centroids)
        
        # Match testing with each current detection
        
        distance_detections = zip(distance[0], current_detections_info)
        
        matched = False
        
        for distance, detection in distance_detections:
            logging.debug(f"Distance to target {target_id}: {distance}")
            
            is_within_tight_distance = distance <= tight_distance
            is_within_max_distance = distance <= max_distance
            color_distance = self._calculate_color_distance(target_id, detection)
            logging.debug(f"Color distance to target {target_id}: {color_distance}")
            color_deficit = np.clip(color_distance / self.max_color_distance, 0.0, 1.0)
            color_confidence = 1.0 - color_deficit  # 0 to 1 confidence
            
            if not is_within_tight_distance:
                if not is_within_max_distance:
                    continue
                elif color_deficit > 0.5:
                    continue
            
            # For target tracking, use the weighted score as well
            score = self._compute_match_score(
                target_id, detection, distance,
                max_distance, tight_distance, frame_width, frame_height
            )
            
            if score >= 1.0:
                logging.debug(f"Target match rejected (score: {score:.3f})")
                continue
            
            matched = True
            logging.debug(f"Target {target_id} matched with detection at score {score:.3f}")
            self._update_confirmed_track(target_id, detection)
            break
        
        if not matched:
            self.disappeared_frames[target_id] += 1
            if self.disappeared_frames[target_id] > self.max_disappeared_frames:
                self._deregister(target_id)
                logging.info(f"Target {target_id} deregistered after {self.max_disappeared_frames}")
                return {}
            else:
                vx, vy = self.velocities.get(target_id, (0, 0))
                self.velocities[target_id] = (vx * self.velocity_decay, vy * self.velocity_decay)
        result = self.tracked_objects.get(target_id)
        if result is None:
            return {}
        logging.debug(f"Returning tracked object {target_id}: {self.tracked_objects[target_id]}")
        return {target_id : self.tracked_objects[target_id]}
             
              
    def _build_predicted_centroids(self, object_ids, frame_width, frame_height):
        """
        Build velocity-predicted centroids for confirmed object IDs.
        Velocity weight is suppressed near frame edges to prevent off-screen
        extrapolation, which destabilises matching for partially visible objects.
        
        :param object_ids: list of confirmed object IDs to build predictions for
        :param frame_width: width of the video frame (for distance normalization)
        :param frame_height: height of the video frame (for distance normalization)
        :return: list of predicted centroid tuples corresponding to the input object IDs
        """
        predicted = []
        for object_id in object_ids:
            cx, cy = self.tracked_objects[object_id]['centroid'] # Centroid of most recent detection for this track
            vx, vy = self.velocities.get(object_id, (0, 0)) # Velocity for this track, default to (0, 0) if not available

            edge_proximity  = min(cx, frame_width - cx, cy, frame_height - cy) # Distance to nearest edge of the frame
            velocity_weight = float(np.clip(edge_proximity / self.edge_margin, 0.0, 1.0)) # Velocity's influence lowers as detections reach the edge to account for partial occlusion.
            
            # Predicted centroid coordinates based on current velocity, clipped to frame bounds
            px = int(np.clip(cx + vx * velocity_weight, 0, frame_width - 1))  
            py = int(np.clip(cy + vy * velocity_weight, 0, frame_height - 1)) 
            predicted.append((px, py))
        return predicted
    
    def _calculate_color_distance(self, object_id, detection_info):
        """
        Calculates and returns the euclidean distance between previous 
        detection and current frame detection color for confirmed tracks.
        
        :param object_id: ID of the confirmed track to compute color distance
        :param detection_info: detection info dict from current frame corresponding to this track
        :return: euclidean distance between previous color and input color as a floating-point number
        """
        
        input_color = detection_info['color']
        prev_color = self.colors[object_id]
        
        return np.linalg.norm(np.array(input_color) - np.array(prev_color))
    
    def _calculate_color_distance_tentative(self, tent_id, detection_info):
        """
        Calculates color distance for tentative tracks (uses detection's own color).
        
        :param tent_id: ID of the tentative track
        :param detection_info: detection info dict from current frame
        :return: euclidean distance between tentative object's color and detection's color
        """
        tent_color = self._tentative_objects[tent_id].get('color', (0, 0, 0))
        detection_color = detection_info.get('color', (0, 0, 0))
        
        return np.linalg.norm(np.array(detection_color) - np.array(tent_color))
    
    def _compute_iou(self, bbox1, bbox2):
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes.
        
        :param bbox1: list of bounding box edges [x1, y1, x2, y2] for current detection
        :param bbox2: list of bounding box edges [x1, y1, x2, y2] for last stored detection for id
        :return: overlap value of two bounding boxes to test against threshold
        
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def _compute_match_score(self, object_id, detection, distance, max_distance, 
                             tight_distance, frame_width, frame_height):
        """
        Compute a weighted matching score (0.0 = perfect match, 1.0 = worst match).
        Combines spatial distance, color similarity, label match, confidence match, and IoU.
        Returns 2.0 (invalid) if hard constraints are violated.
        
        Weights:
        - Spatial distance: 0.35 (most important, normalized to max_distance)
        - Color distance: 0.25
        - Label match: 0.20 (0 if match, 1 if mismatch)
        - Confidence match: 0.15 (0 if similar, 1 if very different)
        - IoU bonus: -0.05 (negative weight, subtracts from score if boxes overlap well)
        
        :param object_id: ID of the confirmed track
        :param detection: detection info dict from current frame
        :param distance: spatial distance already computed
        :param max_distance: max allowed spatial distance
        :param tight_distance: tight distance threshold
        :param frame_width: width of frame (for normalization)
        :param frame_height: height of frame
        :return: float score (lower is better), or 2.0 if invalid match
        """
        # Hard constraint 1: Must be within spatial bounds
        if distance > max_distance:
            return 2.0
        
        # Hard constraint 2: Label must match if both are available
        tracked_label = self.tracked_objects[object_id].get('label')
        detection_label = detection.get('label')
        if tracked_label and detection_label and tracked_label != detection_label:
            logging.debug(f"Label hard constraint violated: {tracked_label} vs {detection_label}")
            return 2.0
        
        # Hard constraint 3: Confidence must be reasonably similar
        tracked_conf = self.tracked_objects[object_id].get('confidence', 0.0)
        detection_conf = detection.get('confidence', 0.0)
        if abs(tracked_conf - detection_conf) > 0.25:
            logging.debug(f"Confidence constraint: {tracked_conf:.2f} vs {detection_conf:.2f}")
            return 2.0
        
        # Start building soft-constraint score
        score = 0.0
        
        # Spatial distance component (0.35 weight, normalized to max_distance)
        spatial_score = min(1.0, distance / max_distance)
        score += spatial_score * 0.35
        
        # Color distance component (0.25 weight)
        color_distance = self._calculate_color_distance(object_id, detection)
        color_deficit = np.clip(color_distance / self.max_color_distance, 0.0, 1.0)
        score += color_deficit * 0.25
        
        # Label match bonus (0.20 weight) - 0 if match, 0.20 if mismatch
        label_score = 0.0 if (tracked_label == detection_label or not tracked_label or not detection_label) else 0.20
        score += label_score
        
        # Confidence match bonus (0.15 weight) - penalize large differences
        conf_diff_norm = min(1.0, abs(tracked_conf - detection_conf) / 0.25)
        score += conf_diff_norm * 0.15
        
        # IoU bonus (negative weight -0.05) - reduces score if boxes overlap well
        tracked_bbox = self.tracked_objects[object_id]['bbox']
        detection_bbox = detection['bbox']
        iou = self._compute_iou(tracked_bbox, detection_bbox)
        score -= max(0.0, iou - 0.1) * 0.05  # Reward overlap > 0.1, penalty if very low
        
        return max(0.0, score)  # Clamp to non-negative

    def _update_confirmed_track(self, object_id, detection_info):
        """
        Update a confirmed track with a matched detection and recompute velocity.
        
        :param object_id: ID of the confirmed track to update
        :param detection_info: detection info dict from the current frame that matched this track
        """
        prev_centroid = self.tracked_objects[object_id]['centroid']
        new_centroid  = detection_info['centroid']
        vx = new_centroid[0] - prev_centroid[0] # Per-frame x-direction velocity (pixels/franme)
        vy = new_centroid[1] - prev_centroid[1] # Per-frame y-direction velocity (pixels/frame)
        self.velocities[object_id] = (vx, vy) # Add or update velocity for this track
        self.colors[object_id] = detection_info['color']
        logging.debug(f"Object {object_id} color: {self.colors[object_id]}")

        if vx != 0 or vy != 0:
            logging.debug(f"Object {object_id} velocity: ({vx}, {vy})")

        self.tracked_objects[object_id]    = detection_info
        self.disappeared_frames[object_id] = 0

    def _age_confirmed(self, matched_rows):
        """
        Age all confirmed tracks not matched this frame.
        Decays velocity and deregisters tracks absent too long.
        matched_rows is a set of row indices into the confirmed ID list.
        
        :param matched_rows: set of row indices corresponding to confirmed tracks that were matched this frame
        """
        for row, object_id in enumerate(list(self.tracked_objects.keys())):
            if row in matched_rows:
                continue
            self.disappeared_frames[object_id] += 1
            if self.disappeared_frames[object_id] > self.max_disappeared_frames:
                self._deregister(object_id)
            else:
                # Decay velocity toward zero so prediction doesn't drift during absence
                vx, vy = self.velocities.get(object_id, (0, 0))
                self.velocities[object_id] = (vx * self.velocity_decay, vy * self.velocity_decay)

    def _age_tentative(self, matched_tent_rows):
        """
        Discard any tentative track not matched this frame.
        Tentative tracks get no second chances — one missed frame and they're gone.
        Real objects persist across frames; single-frame ghosts do not.
        
        :param matched_tent_rows: set of row indices corresponding to tentative tracks that were matched this frame
        """
        for row, tent_id in enumerate(list(self._tentative_objects.keys())):
            if row not in matched_tent_rows:
                del self._tentative_objects[tent_id]
                del self._tentative_hits[tent_id]

    def _register_tentative(self, detection_info):
        """
        Add a new tentative track candidate. Uses negative IDs internally.
        
        :param detection_info: detection info dict from the current frame for this new tentative track
        """
        self._tentative_id_seq -= 1
        tid = self._tentative_id_seq
        self._tentative_objects[tid] = detection_info
        self._tentative_hits[tid] = 1

    def _promote_tentative(self, tent_id):
        """
        Graduate a tentative track to a confirmed track with a stable positive ID.
        
        :param tent_id: ID of the tentative track to promote
        """
        detection_info = self._tentative_objects.pop(tent_id)
        self._tentative_hits.pop(tent_id)
        new_id = self._get_next_id()
        self.tracked_objects[new_id] = detection_info
        self.disappeared_frames[new_id] = 0 # Start with zero disappeared frames since it's just been confirmed
        self.colors[new_id] = detection_info.get('color', (0,0,0))
       
        logging.debug(f"Track ID {new_id} confirmed after {self.hit_streak_required} consecutive matches.")

    def _get_next_id(self): # Get the next unique ID for a new confirmed track
        new_id = self.next_id_counter
        self.next_id_counter += 1
        return new_id

    def _deregister(self, object_id): # Remove a track that has disappeared for too long
        logging.debug(f"Track ID {object_id} deregistered after {self.disappeared_frames[object_id]} missed frames.")
        del self.tracked_objects[object_id]
        del self.disappeared_frames[object_id]
        self.velocities.pop(object_id, None)
        self.colors.pop(object_id, None)
