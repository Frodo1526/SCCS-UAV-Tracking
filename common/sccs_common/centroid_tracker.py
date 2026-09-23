import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import logging

# NOTE: do NOT call logging.basicConfig() at module import time here.
# This module is imported by more than one independent program (drone_air,
# tello_poc); each program's own entry point (main.py) is responsible for
# configuring logging exactly once. A basicConfig() call in a shared library
# module would silently win the process-wide log level/handler for whichever
# program happens to import it first -- this is why telloControl/app.log grew
# to 28+ MB at DEBUG level in the original codebase (see the earlier code
# review). The logging.debug()/info()/warning() calls below use the root
# logger, which the consuming program's own basicConfig() call configures.


class CentroidTracker:
    def __init__(self, max_disappeared=20, tight_distance_ratio=0.05, max_distance_ratio=0.10, hit_streak_required=3, velocity_decay=0.5,
                 edge_margin=20, next_id_counter=1, max_color_distance=15.0, iou_suppresion_thresh=0.6):
        '''
        Initializes a CentroidTracker with empty confirmed/tentative track
        registries and the tunable parameters used by update_all_detections()
        and update_target() for matching, scoring, and aging tracks.

        Args:
            max_disappeared (int): Number of consecutive unmatched frames a
                confirmed track may accumulate before it is deregistered.
                Defaults to 20.
            tight_distance_ratio (float): Fraction of frame width defining the
                "tight" distance threshold; within this distance, update_target()
                accepts a match even if the color deficit is otherwise borderline.
                Defaults to 0.05.
            max_distance_ratio (float): Fraction of frame width defining the
                maximum spatial distance between a predicted centroid and a
                candidate detection for the pair to be considered a possible
                match at all. Defaults to 0.10.
            hit_streak_required (int): Number of consecutive frames a tentative
                track must be matched before it is promoted to a confirmed
                track with a stable ID. Defaults to 3.
            velocity_decay (float): Per-frame multiplicative decay applied to a
                track's stored velocity while it goes unmatched, e.g. 0.5 halves
                the velocity each unmatched frame to limit prediction drift
                during occlusion. Defaults to 0.5.
            edge_margin (int): Distance in pixels from the nearest frame edge
                within which a track's velocity-based position prediction is
                progressively scaled down, so predictions don't extrapolate
                off-screen near boundaries. Defaults to 20.
            next_id_counter (int): Starting value of the incrementing counter
                used to assign IDs to newly confirmed (promoted) tracks.
                Defaults to 1.
            max_color_distance (float): Euclidean color-distance value used to
                normalize the color component of the match score; a distance at
                or above this value is treated as a complete color mismatch
                (deficit of 1.0). Defaults to 15.0.
            iou_suppresion_thresh (float): IoU threshold above which an
                unmatched detection is treated as a duplicate of an existing
                confirmed track's bounding box and is excluded from becoming a
                new tentative track. Defaults to 0.6.

        Returns:
            None: This function does not return a value.
        '''
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
        '''
        Runs one full tracking update for every detection in the current frame,
        matching detections against confirmed tracks first, then tentative
        tracks, and registering any leftovers as new tentative tracks.

        Matching proceeds in two Hungarian-algorithm passes:
        1. Confirmed tracks vs. all detections, scored by
            _compute_match_score (spatial distance, color, label, confidence,
            IoU). Matches with a cost below 1.0 are accepted; matched confirmed
            tracks are updated via _update_confirmed_track, and unmatched
            confirmed tracks are aged via _age_confirmed.
        2. Remaining unmatched detections (after discarding any that overlap
            an existing confirmed track's bbox above iou_suppresion_thresh)
            vs. tentative tracks, scored by a simplified spatial+color cost.
            Matches below 1.0 increment the tentative track's hit streak and
            promote it to confirmed once hit_streak_required is reached;
            unmatched tentative tracks are discarded via _age_tentative.
        3. Any detection still unmatched after both passes is registered as a
            brand-new tentative track via _register_tentative.

        If current_detections_info is empty, or contains non-finite (NaN/Inf)
        centroid values, all confirmed and tentative tracks are aged/discarded
        as if nothing matched, and no new tracks are registered.

        Args:
            current_detections_info (list[dict]): Detections for this frame.
                Each dict must contain at least 'centroid' (x, y) and 'bbox'
                (x1, y1, x2, y2); 'label', 'confidence', and 'color' are used
                when present for scoring.
            frame_width (int): Width of the video frame, used to normalize
                distance thresholds.
            frame_height (int): Height of the video frame, used for
                velocity-prediction edge suppression.

        Returns:
            dict: self.tracked_objects, an OrderedDict mapping confirmed track
                IDs to their most recent detection info dict. Tentative
                (unconfirmed) tracks are never included in the return value.
        '''
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
        '''
        Updates a single confirmed track (the active tracking target) against
        the current frame's detections, without evaluating matches for any
        other confirmed or tentative track. Used while actively tracking a
        target to reduce per-frame computation versus update_all_detections().

        Behavior:
        - If current_detections_info is empty, the target's disappeared-frame
            count is incremented (and the track deregistered if it exceeds
            max_disappeared_frames) or its velocity is decayed, and {} is
            returned.
        - If target_id is not present in self.tracked_objects, the target is
            logged as not found and {} is returned (no fallback to matching
            against all detections is performed).
        - Otherwise, a velocity-predicted centroid for the target is computed
            (edge-margin-suppressed, as in _build_predicted_centroids) and
            compared against every detection in order. A detection is accepted
            as a match if it is within tight_distance, or within max_distance
            with a color deficit no greater than 0.5, and the full weighted
            match score (_compute_match_score) is below 1.0. The first
            detection satisfying these conditions is used to update the track
            via _update_confirmed_track, and the search stops.
        - If no detection matches, the target's disappeared-frame count is
            incremented; the track is deregistered if this exceeds
            max_disappeared_frames (returning {}), otherwise its velocity is
            decayed.

        Args:
            current_detections_info (list[dict]): Detections for this frame,
                each with at least 'centroid', 'bbox', and optionally 'color',
                'label', 'confidence'.
            target_id: ID of the confirmed track to update. Must be a key of
                self.tracked_objects for a match attempt to be made.
            frame_width (int): Width of the video frame, used to normalize the
                distance threshold and edge-proximity calculation.
            frame_height (int): Height of the video frame, used for the
                edge-proximity calculation.

        Returns:
            dict: {target_id: <detection info dict>} if the target track is
                still confirmed after this update (matched or not-yet-expired).
            dict: {} (empty) if there were no detections this frame, if
                target_id is not a known confirmed track, or if the target was
                deregistered this call after exceeding max_disappeared_frames.
        '''
        
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
        '''
        Computes a velocity-extrapolated centroid for each given confirmed
        track ID, used as the "expected position this frame" against which
        incoming detections are matched.

        For each track, its last known centroid is advanced by its stored
        velocity, but the velocity's influence is scaled down as the centroid
        nears any frame edge (down to zero at the edge itself, full strength
        once at least edge_margin pixels from every edge). This prevents the
        prediction from extrapolating a track off-screen when it is partially
        occluded by the frame boundary. The result is clipped to stay within
        the frame.

        Args:
            object_ids (list): Confirmed track IDs (keys of self.tracked_objects)
                to build predictions for, in the order predictions should be
                returned.
            frame_width (int): Width of the video frame, used to clip the
                predicted x-coordinate and compute edge proximity.
            frame_height (int): Height of the video frame, used to clip the
                predicted y-coordinate and compute edge proximity.

        Returns:
            list[tuple[int, int]]: Predicted (x, y) centroid for each ID in
                object_ids, in the same order.
        '''
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
        '''
        Computes the Euclidean distance between a confirmed track's
        last-recorded color and a candidate detection's color, used as the
        color component of the match score.

        Args:
            object_id: ID of the confirmed track; must be a key of self.colors.
            detection_info (dict): Detection info dict for the current frame
                candidate; must contain a 'color' value.

        Returns:
            float (numpy.float64): Euclidean distance between the track's
                stored color and the detection's color.

        Raises:
            KeyError: If object_id is not present in self.colors.
            KeyError: If detection_info does not contain a 'color' key.
        '''
        
        input_color = detection_info['color']
        prev_color = self.colors[object_id]
        
        return np.linalg.norm(np.array(input_color) - np.array(prev_color))
    
    def _calculate_color_distance_tentative(self, tent_id, detection_info):
        '''
        Computes the Euclidean color distance between a tentative track's
        stored color and a candidate detection's color, used when scoring
        tentative-track matches in update_all_detections().

        Unlike _calculate_color_distance, this falls back to (0, 0, 0) for
        either side if a 'color' value is missing, rather than raising.

        Args:
            tent_id: ID of the tentative track (key of self._tentative_objects).
            detection_info (dict): Detection info dict for the current frame
                candidate.

        Returns:
            float (numpy.float64): Euclidean distance between the tentative
                track's color and the detection's color.
        '''
        tent_color = self._tentative_objects[tent_id].get('color', (0, 0, 0))
        detection_color = detection_info.get('color', (0, 0, 0))
        
        return np.linalg.norm(np.array(detection_color) - np.array(tent_color))
    
    def _compute_iou(self, bbox1, bbox2):
        '''
        Computes the Intersection over Union (IoU) of two axis-aligned
        bounding boxes.

        Args:
            bbox1 (Sequence[float]): First bounding box as (x1, y1, x2, y2).
            bbox2 (Sequence[float]): Second bounding box as (x1, y1, x2, y2).

        Returns:
            float: 0.0 if the boxes do not overlap. Otherwise the overlap
                ratio in [0.0, 1.0], i.e. intersection area divided by union
                area (0.0 if the union area is also zero).
        '''
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
    
    def _compute_match_score(self, object_id, detection, distance, max_distance, tight_distance, frame_width, frame_height):
        '''
        Computes a weighted match-quality score between a confirmed track and
        a candidate detection, combining spatial distance, color similarity,
        label agreement, confidence similarity, and bounding-box overlap.
        Lower is better; 0.0 is a perfect match.

        Two hard constraints are checked first and cause an immediate invalid
        result (2.0):
        1. distance exceeds max_distance.
        2. Both the track and the detection have a non-empty 'label' and
            they differ.
        3. The track's and detection's 'confidence' values (default 0.0 if
            absent) differ by more than 0.25.

        If all hard constraints pass, the soft score is the sum of:
        - Spatial distance, normalized to max_distance, weight 0.35.
        - Color distance (via _calculate_color_distance), normalized to
            self.max_color_distance, weight 0.25.
        - Label mismatch penalty, weight 0.20 (0 if either label is empty or
            they match, 0.20 if both present and differ).
        - Confidence-difference penalty, normalized to a 0.25 span, weight
            0.15.
        - An IoU bonus (via _compute_iou) that subtracts up to 0.05 from the
            score when the boxes overlap more than 0.1 IoU.
        The final score is clamped to be non-negative.

        Args:
            object_id: ID of the confirmed track being scored against.
            detection (dict): Candidate detection info dict for the current
                frame; expected to contain 'bbox' and optionally 'label',
                'confidence'.
            distance (float): Pre-computed spatial distance between the
                track's predicted centroid and the detection's centroid.
            max_distance (float): Maximum spatial distance allowed for any
                match (hard constraint threshold).
            tight_distance (float): Unused within this method; accepted for a
                consistent call signature with other matching helpers.
            frame_width (int): Unused within this method; accepted for a
                consistent call signature with other matching helpers.
            frame_height (int): Unused within this method; accepted for a
                consistent call signature with other matching helpers.

        Returns:
            float: 2.0 if any hard constraint is violated (an invalid match).
            float: A non-negative soft-constraint score, normally in roughly
                [0.0, 1.0], where lower indicates a better match.
        '''
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
        '''
        Applies a matched detection to a confirmed track: recomputes and stores
        the per-frame velocity from the change in centroid, updates the stored
        color, replaces the track's detection info with the new detection, and
        resets its disappeared-frame counter to 0.

        Args:
            object_id: ID of the confirmed track to update; must already be a
                key of self.tracked_objects.
            detection_info (dict): The detection info dict that matched this
                track this frame; must contain 'centroid' and 'color'.

        Returns:
            None: This function does not return a value. It mutates
                self.velocities, self.colors, self.tracked_objects, and
                self.disappeared_frames in place.

        Raises:
            KeyError: If object_id is not already present in
                self.tracked_objects.
            KeyError: If detection_info lacks a 'centroid' or 'color' key.
        '''
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
        '''
        Ages every confirmed track that was not matched this frame: increments
        its disappeared-frame count, deregisters it if that count exceeds
        max_disappeared_frames, and otherwise decays its stored velocity toward
        zero so predictions don't keep drifting while the track is absent.

        Args:
            matched_rows (set[int]): Row indices (positions in the current
                iteration order of self.tracked_objects.keys()) of confirmed
                tracks that WERE matched this frame and should be skipped.

        Returns:
            None: This function does not return a value. It mutates
                self.disappeared_frames and self.velocities in place, and may
                remove entries via _deregister.
        '''
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
        '''
        Discards every tentative track that was not matched this frame.
        Tentative tracks have no aging grace period: a single missed frame
        causes immediate removal, on the premise that real objects are matched
        on consecutive frames while single-frame false positives are not.

        Args:
            matched_tent_rows (set[int]): Row indices (positions in the current
                iteration order of self._tentative_objects.keys()) of tentative
                tracks that WERE matched this frame and should be kept.

        Returns:
            None: This function does not return a value. It removes entries
                from self._tentative_objects and self._tentative_hits in place.
        '''
        for row, tent_id in enumerate(list(self._tentative_objects.keys())):
            if row not in matched_tent_rows:
                del self._tentative_objects[tent_id]
                del self._tentative_hits[tent_id]

    def _register_tentative(self, detection_info):
        '''
        Registers a new tentative (unconfirmed) track for a detection that did
        not match any existing confirmed or tentative track. Tentative tracks
        use a negative, decrementing internal ID sequence so they can never
        collide with the positive IDs used for confirmed tracks.

        Args:
            detection_info (dict): Detection info dict for the current frame
                that will seed the new tentative track.

        Returns:
            None: This function does not return a value. It adds an entry to
                self._tentative_objects (with hit count 1 in
                self._tentative_hits) and decrements self._tentative_id_seq.
        '''
        self._tentative_id_seq -= 1
        tid = self._tentative_id_seq
        self._tentative_objects[tid] = detection_info
        self._tentative_hits[tid] = 1

    def _promote_tentative(self, tent_id):
        '''
        Promotes a tentative track that has reached hit_streak_required
        consecutive matches into a confirmed track with a new, stable positive
        ID from _get_next_id(). The tentative track's bookkeeping entries are
        removed as part of the promotion.

        Args:
            tent_id: ID of the tentative track to promote; must be a key of
                self._tentative_objects.

        Returns:
            None: This function does not return a value. It removes tent_id
                from self._tentative_objects/self._tentative_hits and adds a
                new entry to self.tracked_objects, self.disappeared_frames,
                and self.colors.

        Raises:
            KeyError: If tent_id is not present in self._tentative_objects or
                self._tentative_hits.
        '''
        detection_info = self._tentative_objects.pop(tent_id)
        self._tentative_hits.pop(tent_id)
        new_id = self._get_next_id()
        self.tracked_objects[new_id] = detection_info
        self.disappeared_frames[new_id] = 0 # Start with zero disappeared frames since it's just been confirmed
        self.colors[new_id] = detection_info.get('color', (0,0,0))
       
        logging.debug(f"Track ID {new_id} confirmed after {self.hit_streak_required} consecutive matches.")

    def _get_next_id(self):
        '''
        Returns the next unique ID to assign to a newly confirmed track, and
        advances the internal counter so the same ID is never issued twice.

        Returns:
            int: The next available confirmed-track ID (the pre-increment
                value of self.next_id_counter).
        '''
        new_id = self.next_id_counter
        self.next_id_counter += 1
        return new_id

    def _deregister(self, object_id):
        '''
        Removes a confirmed track from all tracker state because it has gone
        unmatched for longer than max_disappeared_frames.

        Args:
            object_id: ID of the confirmed track to remove; must be a key of
                self.tracked_objects and self.disappeared_frames.

        Returns:
            None: This function does not return a value. It deletes object_id
                from self.tracked_objects and self.disappeared_frames, and
                removes it (if present) from self.velocities and self.colors.

        Raises:
            KeyError: If object_id is not present in self.tracked_objects or
                self.disappeared_frames.
        '''
        logging.debug(f"Track ID {object_id} deregistered after {self.disappeared_frames[object_id]} missed frames.")
        del self.tracked_objects[object_id]
        del self.disappeared_frames[object_id]
        self.velocities.pop(object_id, None)
        self.colors.pop(object_id, None)