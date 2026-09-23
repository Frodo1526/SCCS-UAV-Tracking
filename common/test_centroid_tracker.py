"""Basic regression tests for sccs_common.centroid_tracker.CentroidTracker.

These are intentionally simple, deterministic, hardware-independent tests --
no camera, no GStreamer, no drone required. This is the highest-value place
in the whole project to have test coverage, since CentroidTracker's matching
logic is exactly the kind of thing that's easy to silently break while
refactoring (see the original code review's "Testing" section).
"""

from sccs_common.centroid_tracker import CentroidTracker


def make_detection(cx, cy, w=40, h=40, label="car", confidence=0.8, color=(100, 100, 100)):
    '''
    Builds a synthetic detection info dict centered at (cx, cy) for use in
    CentroidTracker tests, matching the dict shape the tracker expects
    ('bbox', 'centroid', 'label', 'confidence', 'color').

    Args:
        cx (int): Centroid x-coordinate of the detection.
        cy (int): Centroid y-coordinate of the detection.
        w (int): Bounding-box width, used to derive 'bbox' around (cx, cy).
            Defaults to 40.
        h (int): Bounding-box height, used to derive 'bbox' around (cx, cy).
            Defaults to 40.
        label (str): Detection class label. Defaults to "car".
        confidence (float): Detection confidence score. Defaults to 0.8.
        color (tuple): Sampled color for the detection. Defaults to
            (100, 100, 100).

    Returns:
        dict: A detection info dict with keys 'bbox', 'centroid', 'label',
            'confidence', and 'color'.
    '''
    return {
        "bbox": (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2),
        "centroid": (cx, cy),
        "label": label,
        "confidence": confidence,
        "color": color,
    }

def test_new_detection_starts_tentative_not_confirmed():
    '''
    Verifies that a single detection on the first frame is registered only
    as a tentative track and is not yet present in the confirmed results
    returned by update_all_detections().

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via assert statements.
    '''
    tracker = CentroidTracker(hit_streak_required=3)
    detections = [make_detection(100, 100)]

    result = tracker.update_all_detections(detections, frame_width=640, frame_height=480)

    # A single frame is not enough to confirm a new track.
    assert result == {}
    assert len(tracker._tentative_objects) == 1

def test_track_promotes_to_confirmed_after_hit_streak():
    '''
    Verifies that a detection appearing at roughly the same position for
    hit_streak_required consecutive frames is promoted from a tentative
    track to a confirmed track, and that the confirmed result preserves
    the detection's centroid.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via assert statements.
    '''
    tracker = CentroidTracker(hit_streak_required=3, tight_distance_ratio=0.2, max_distance_ratio=0.4)

    # Same object, roughly the same place, for hit_streak_required consecutive frames.
    result = {}
    for _ in range(3):
        detections = [make_detection(100, 100)]
        result = tracker.update_all_detections(detections, frame_width=640, frame_height=480)

    assert len(result) == 1
    (object_id, info), = result.items()
    assert info["centroid"] == (100, 100)

def test_confirmed_track_deregisters_after_max_disappeared():
    '''
    Verifies that a confirmed track is removed from tracked_objects once
    it has been fed more consecutive empty-detection frames than
    max_disappeared allows.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via assert statements.
    '''
    tracker = CentroidTracker(hit_streak_required=1, max_disappeared=2)

    # Confirm a track.
    tracker.update_all_detections([make_detection(100, 100)], 640, 480)
    assert len(tracker.tracked_objects) == 1

    # Then feed empty frames until it should be dropped.
    tracker.update_all_detections([], 640, 480)
    tracker.update_all_detections([], 640, 480)
    tracker.update_all_detections([], 640, 480)

    assert len(tracker.tracked_objects) == 0

def test_update_all_detections_ignores_nan_centroids():
    '''
    Verifies that update_all_detections() does not raise and does not
    register any track when given a detection whose centroid contains a
    NaN value.

    Returns:
        None: This is a pytest-style test function; it does not return a
            value, and signals failure via assert statements.
    '''
    tracker = CentroidTracker()
    bad_detection = make_detection(100, 100)
    bad_detection["centroid"] = (float("nan"), 100)

    # Should not raise, and should not register the bad detection.
    result = tracker.update_all_detections([bad_detection], 640, 480)
    assert result == {}
    assert len(tracker._tentative_objects) == 0

def test_update_target_returns_empty_dict_for_unknown_target_id():
    """
    Documents actual behavior: update_target does NOT fall back to
    update_all_detections for an unrecognized target_id -- it returns {}.
    See the documentation-vs-code comparison: drone_software_documentation.pdf
    claims a fallback to update_all_detections that the code does not
    implement. This test pins down the real (documented-here) behavior.
    """
    tracker = CentroidTracker()
    result = tracker.update_target([make_detection(100, 100)], target_id=999, frame_width=640, frame_height=480)
    assert result == {}
