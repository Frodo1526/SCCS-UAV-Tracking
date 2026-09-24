import threading

class AppState:
    """Manages shared state between GStreamer and UI threads"""
    def __init__(self):
        '''
        Constructs the shared mutable state used across the ground_station
        subsystem's threads (GStreamer video receiver, telemetry data
        receiver, and the Pygame UI/main loop): the most recent decoded video
        frame, the most recent tracking-data list, the last mouse click
        position, the currently selected target ID, stop/error events for the
        background threads, and a set of threading.Lock objects guarding each
        piece of shared state.

        Returns:
            None: This function does not return a value.
        '''
        self.pygame_frame = None                        # The most recent numpy frame
        self.tracking_data = []                         # The most recent tracking data
        self.last_click_pos = None                      # (x, y) tuple of the last click
        self.current_target_id = None                   # The ID we are currently tracking
        self.gst_error_event = threading.Event()        # Set if GStreamer fails
        self.data_thread_stop_event = threading.Event() # Set to signal data thread to stop
       
        # Thread locks for synchronization
        self.frame_lock = threading.Lock()
        self.tracking_data_lock = threading.Lock()
        self.click_lock = threading.Lock()
        self.target_lock = threading.Lock()