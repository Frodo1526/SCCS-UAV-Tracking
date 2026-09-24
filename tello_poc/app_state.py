from collections import OrderedDict
import threading
import time

from common.sccs_common.centroid_tracker import CentroidTracker

class AppState:
    def __init__(self):
        '''
        Constructs the shared mutable state for the Tello proof-of-concept
        subsystem: a CentroidTracker instance configured for this subsystem,
        the selected target ID, current drone state machine value, the latest
        video frame and its timestamp, the latest tracked-object snapshot,
        HOVERING/RETURNING timer timestamps, the process start time, sets used
        to track currently- and recently-pressed manual-control keys, the
        current command display string, a set of threading.Lock objects
        guarding each piece of shared state, stop events for the main and
        control loops, whether the drone is currently airborne, and the last
        known battery percentage.

        Returns:
            None: This function does not return a value.
        '''
        self.tracker = CentroidTracker(
            max_disappeared=20,
            tight_distance_ratio=0.08,
            max_distance_ratio=0.15,
            hit_streak_required=3,
            velocity_decay=0.5,
            edge_margin=20,
            next_id_counter=1,
            max_color_distance=20.0,
            iou_suppresion_thresh=0.6
        )
        self.target_id = None
        self.drone_state = "MANUAL"
        
        self.frame = None
        self.last_frame_time = 0.0
        self.tracked = OrderedDict()
        self.hover_lost_time = None
        self.hold_start_time = None
        self.start_time = time.time()
        
        self.pressed_keys = set() # Track currently pressed keys for manual control
        self.key_ages = {} # Track frames since each key was last pressed
        
        self.current_command_str = "" # Current command values for display
        
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.target_lock = threading.Lock() 
        self.tracker_lock = threading.Lock()
        self.battery_lock = threading.Lock()
        self.keys_lock = threading.Lock() # For manual control key tracking
        self.command_lock = threading.Lock() # For command value display
        
        self.stop_event = threading.Event()
        self.control_stop = threading.Event()
        self.airborne = False
        self.battery = 100