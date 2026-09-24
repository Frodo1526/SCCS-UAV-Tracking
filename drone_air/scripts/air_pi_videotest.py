#Stripped down version of air_pi for video-only testing
#Removes DroneCommandController, Centroid Tracker, AppState, etc.
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import time
import threading

# -----------------------------------------------------------------------------------------------
# Network Configuration
# -----------------------------------------------------------------------------------------------
GROUND_STATION_IP = "10.5.0.1"
VIDEO_STREAM_PORT = 5602

# -----------------------------------------------------------------------------------------------
# Test Constants
# -----------------------------------------------------------------------------------------------
TARGET_LOST_HOVER_TIMEOUT = 5.0     
HOLD_BEFORE_MANUAL_DURATION = 2.0   
STATE_CYCLE_INTERVAL = 0.1         

# -----------------------------------------------------------------------------------------------
# State Test
# -----------------------------------------------------------------------------------------------
class TestState:                            
    def __init__(self):
        '''
        Initializes the stripped-down test state used by the video-only smoke
        test: the current simulated drone state (starting at "MANUAL"), the
        simulated target ID (starting unset), a lock guarding both, and a stop
        event for the state-machine thread.

        Returns:
            None: This function does not return a value.
        '''
        self.drone_state = "MANUAL"         
        self.target_id = None               
        self.state_lock = threading.Lock()  
        self.stop_event = threading.Event() 

# -----------------------------------------------------------------------------------------------
#  State Machine Thread
# -----------------------------------------------------------------------------------------------
def run_state_machine(test_state):
    '''
    Runs a simulated MANUAL -> TRACKING -> HOVERING -> RETURNING -> MANUAL
    state machine on a fixed polling interval (STATE_CYCLE_INTERVAL),
    printing the current state each cycle, to validate the state-transition
    logic independent of any real tracker or drone connection. Two daemon
    helper threads are started to simulate operator actions: one selects a
    target after a delay (simulate_target_selection) and one clears it
    after a longer delay (simulate_target_lost), driving the state machine
    through its full cycle once.

    State transition rules mirror the real air-side control loop:
      - MANUAL -> TRACKING when a target_id becomes set.
      - TRACKING -> MANUAL if target_id is cleared, or -> HOVERING if the
        target is otherwise no longer found.
      - HOVERING -> TRACKING if the target reappears, or -> RETURNING after
        TARGET_LOST_HOVER_TIMEOUT seconds elapse.
      - RETURNING -> MANUAL (clearing target_id) after
        HOLD_BEFORE_MANUAL_DURATION seconds elapse.

    Exceptions raised inside the loop body are caught, logged to stdout,
    and the loop continues after a 1-second pause, unless test_state's
    stop_event has been set.

    Args:
        test_state (TestState): Shared state object holding the current
            simulated drone_state, target_id, their lock, and the loop's
            stop_event.

    Returns:
        None: This function does not return a value; it runs until
            test_state.stop_event is set.
    '''        
    hover_lost_time = None
    hold_start_time = None

    print("State machine running. Simulating state transitions...")
    print("States will cycle automatically to validate logic.\n")

    def simulate_target_selection():
        '''
        Waits 3 seconds, then, if the simulated drone is still in "MANUAL"
        state, sets test_state.target_id to 1 to simulate an operator
        selecting a target and prints a notice.

        Returns:
            None: This function does not return a value; it mutates
                test_state.target_id under test_state.state_lock.
        '''
        time.sleep(3)
        with test_state.state_lock:
            if test_state.drone_state == "MANUAL":
                test_state.target_id = 1
                print("\n[SIM] Target ID 1 selected by operator.")

    def simulate_target_lost():
        '''
        Waits 8 seconds, then, if the simulated drone is in "TRACKING" state,
        clears test_state.target_id to simulate the tracker losing the target
        and prints a notice.

        Returns:
            None: This function does not return a value; it mutates
                test_state.target_id under test_state.state_lock.
        '''
        time.sleep(8)
        with test_state.state_lock:
            if test_state.drone_state == "TRACKING":
                test_state.target_id = None
                print("\n[SIM] Target lost (removed from tracker).")

    threading.Thread(target=simulate_target_selection, daemon=True).start()
    threading.Thread(target=simulate_target_lost, daemon=True).start()

    while not test_state.stop_event.is_set():
        try:
            with test_state.state_lock:
                current_state = test_state.drone_state
                target_id = test_state.target_id
            if current_state == "MANUAL":
                print("STATE: MANUAL | Waiting for target selection...")      
                if target_id is not None:
                    print("\n[MANUAL] Target selected. Engaging offboard mode.")
                    print("[MANUAL] → TRACKING")
                    hover_lost_time = None
                    with test_state.state_lock:
                        test_state.drone_state = "TRACKING"

            elif current_state == "TRACKING":
                target_found = target_id is not None
                print(f"STATE: TRACKING | Target = {target_id}")                             
                if target_id is None:
                    print("\n[TRACKING] Target cleared by operator. Returning to MANUAL.")
                    with test_state.state_lock:
                        test_state.drone_state = "MANUAL"
                elif not target_found:
                    print("\n[TRACKING] Target lost. Transitioning to HOVERING.")
                    hover_lost_time = time.time()
                    with test_state.state_lock:
                        test_state.drone_state = "HOVERING"
                else:
                    print(f"  [TRACKING] COMMAND: FWD=0.50m/s | UP=0.10m/s | YAW=1.20deg/s        ", end='\r')

            elif current_state == "HOVERING":
                elapsed = time.time() - hover_lost_time
                remaining = TARGET_LOST_HOVER_TIMEOUT - elapsed
                print("STATE: HOVERING | Target Lost")                                                
                if target_id is not None:
                    print("\n[HOVERING] Target reacquired. Resuming TRACKING.")
                    hover_lost_time = None
                    with test_state.state_lock:
                        test_state.drone_state = "TRACKING"
                elif elapsed >= TARGET_LOST_HOVER_TIMEOUT:
                    print("\n[HOVERING] Timeout elapsed. Transitioning to RETURNING.")
                    hold_start_time = time.time()
                    with test_state.state_lock:
                        test_state.drone_state = "RETURNING"
                else:
                    print(f"  [HOVERING] Holding position... ({remaining:.1f}s)        ", end='\r')

            elif current_state == "RETURNING":
                elapsed = time.time() - hold_start_time
                remaining = HOLD_BEFORE_MANUAL_DURATION - elapsed
                print("STATE: RETURNING")                                      
                if elapsed >= HOLD_BEFORE_MANUAL_DURATION:
                    print("\n[RETURNING] Hold complete. Stopping offboard. Returning to MANUAL.")
                    with test_state.state_lock:
                        test_state.target_id = None
                        test_state.drone_state = "MANUAL"
                    hold_start_time = None
                else:
                    print(f"  [RETURNING] Holding position... ({remaining:.1f}s)        ", end='\r')

            time.sleep(STATE_CYCLE_INTERVAL)

        except Exception as e:
            if not test_state.stop_event.is_set():
                print(f"State machine error: {e}")
                time.sleep(1)

# -----------------------------------------------------------------------------------------------
# GStreamer Pipeline
# -----------------------------------------------------------------------------------------------
def run_gstreamer(main_loop):
    '''
    Builds and runs a minimal GStreamer pipeline that captures from the
    Pi camera at 1280x720 and streams H.264-over-RTP to the ground station
    over UDP, for testing video delivery in isolation from AI inference or
    drone control. Blocks running the given GLib main loop once the
    pipeline reaches the PLAYING state.

    A nested `on_message` bus-message handler prints pipeline state
    changes, quits main_loop on ERROR or EOS.

    Args:
        main_loop (GLib.MainLoop): The GLib main loop to run once the
            pipeline is started; also quit from within the message
            handler on error/EOS.

    Returns:
        None: This function does not return a value.

    Raises:
        SystemExit: If setting the pipeline to the PLAYING state fails
            (via sys.exit(1)).
    '''
    pipeline_str = (
        "libcamerasrc ! "
        "video/x-raw,width=1280,height=720,framerate=30/1,format=NV12 ! "
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency bitrate=1500 speed-preset=superfast key-int-max=15 ! "
        "h264parse ! "
        "rtph264pay config-interval=10 pt=96 ! "
        f"udpsink host={GROUND_STATION_IP} port={VIDEO_STREAM_PORT} sync=false async=false"
    ) #linux command that runs gStreamer pipeline on Raspberry Pi 5

    print(f"Pipeline:\n{pipeline_str}\n") #Printing the pipeline command string for confirmation.

    pipeline = None
    try:
        pipeline = Gst.parse_launch(pipeline_str)
        bus = pipeline.get_bus()
        bus.add_signal_watch()

        def on_message(bus, message):
            '''
            GStreamer bus message handler for the smoke-test pipeline. Prints and
            quits main_loop on an ERROR or EOS message; prints pipeline state
            transitions for STATE_CHANGED messages originating from the pipeline
            itself.

            Args:
                bus (Gst.Bus): The message bus that emitted the message (unused
                    beyond signal-connection identity).
                message (Gst.Message): The bus message to handle.

            Returns:
                None: This function does not return a value; it may call
                    main_loop.quit() as a side effect.
            '''
            t = message.type
            if t == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"GStreamer ERROR: {err} | Debug: {debug}")
                main_loop.quit()
            elif t == Gst.MessageType.EOS:
                print("GStreamer: End of stream")
                main_loop.quit()
            elif t == Gst.MessageType.STATE_CHANGED:
                if message.src == pipeline:
                    old, new, _ = message.parse_state_changed()
                    print(f"Pipeline state: {old.value_nick} → {new.value_nick}")

        bus.connect("message", on_message) #Connect message signal to on_message callback function
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR: Could not set pipeline to PLAYING state.")
            sys.exit(1)
        print("GStreamer pipeline running — streaming video to ground station...")
        main_loop.run()

    except Exception as e:
        print(f"GStreamer error: {e}")
    finally:
        print("Stopping pipeline...")
        if pipeline:
            pipeline.set_state(Gst.State.NULL)

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
def main():
    '''
    Entry point for the video-only smoke test. Initializes GStreamer and a
    GLib main loop, starts the simulated state-machine thread
    (run_state_machine) and the GStreamer streaming thread (run_gstreamer)
    as daemons, then polls once per second until the GStreamer thread dies
    or a KeyboardInterrupt occurs. On shutdown, signals the state machine
    to stop, quits the main loop if running, joins both threads with a
    3-second timeout, and exits.

    Returns:
        None: This function does not return; it calls sys.exit(0) at the
            end of its finally block.
    '''
    Gst.init(None) #initialize gStreamer library
    main_loop = GLib.MainLoop() #loop manager

    print("=== Air Pi — Video Smoke Test ===")
    print(f"Streaming to: {GROUND_STATION_IP}:{VIDEO_STREAM_PORT}")
    print("Press Ctrl+C to stop.\n")

    test_state = TestState()  
    sm_thread = threading.Thread(target=run_state_machine, args=(test_state,), daemon=True)
    sm_thread.start()
    gst_thread = threading.Thread(target=run_gstreamer, args=(main_loop,), daemon=True)
    gst_thread.start()

    try:
        while True:
            if not gst_thread.is_alive():
                print("GStreamer thread exited.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutdown requested (Ctrl+C)...")
    finally:
        test_state.stop_event.set()
        if main_loop.is_running():
            main_loop.quit()
        sm_thread.join(timeout=3)
        gst_thread.join(timeout=3)
        print("Air Pi stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()