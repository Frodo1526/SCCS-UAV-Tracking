import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import threading
import time
from pathlib import Path

from app_state import AppState
from command_control import DroneCommandController
from hailo_inference import find_best_model
from vid_cmd_data import run_gstreamer, run_command_server, run_drone_control, load_config

_CFG = load_config()
_NET = _CFG.get("network", {})
GROUND_STATION_IP = _NET.get("ground_ip", "10.5.0.1")
VIDEO_STREAM_PORT = int(_NET.get("video_port", 5602))
DATA_PORT = int(_NET.get("data_port", 5601))
COMMAND_PORT = int(_NET.get("command_port", 5603))      


# -----------------------------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------------------------
def main():
    '''
    Entry point for the drone_air (Raspberry Pi / Hailo) subsystem.
    Initializes GStreamer and a GLib main loop, constructs the shared
    AppState and DroneCommandController, locates the Hailo post-process
    shared library and the best available detection model (via
    find_best_model()), builds the GStreamer pipeline string (camera
    source split via `tee` into an AI-inference branch feeding Hailo
    detection, and a network-streaming branch encoding H.264 over RTP to
    the ground station), and starts three daemon threads: one running the
    GStreamer pipeline (run_gstreamer), one running the TCP command server
    (run_command_server), and one running the MAVSDK drone control loop
    (run_drone_control). The main thread then polls
    app_state.gst_error_event once per second until it is set or a
    KeyboardInterrupt occurs, at which point all stop events are set, the
    GLib main loop is quit, threads are joined with a 3-second timeout
    each, the UDP telemetry socket is closed, and the process exits.

    Returns:
        None: This function does not return; it calls sys.exit(0) at the
            end of its finally block.

    Raises:
        SystemExit: Raised (via sys.exit(1)) if the Hailo post-process
            shared library is not found at the expected path.
    '''
    Gst.init(None)
    main_loop = GLib.MainLoop()
    print("Initializing drone object tracker...")
    app_state = AppState()
    drone_controller = DroneCommandController()

    # --- Post-process library location ---
    postprocess_so = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"
    if not Path(postprocess_so).exists():
        print(f"ERROR: Post-process library not found at {postprocess_so}")
        print("Please check your Hailo TAPPAS installation")
        sys.exit(1)
    print(f"Using post-process library: {postprocess_so}")

    # --- Find model ---
    script_dir = Path(__file__).resolve().parent  # Define script directory
    model_path = find_best_model()

    def QUEUE(name, max_size_buffers=3, leaky="downstream"):
        '''
        Builds a GStreamer `queue` element description string configured to
        drop buffers (rather than block the pipeline) once its buffer limit is
        reached, used to keep the AI and network branches of the pipeline from
        stalling each other.

        Args:
            name (str): Value for the queue element's `name` property, used to
                identify it in the pipeline description.
            max_size_buffers (int): Value for the queue's `max-size-buffers`
                property; `max-size-bytes` and `max-size-time` are always set
                to 0 (unlimited) so only the buffer count limit applies.
                Defaults to 3.
            leaky (str): Value for the queue's `leaky` property, controlling
                which end drops buffers when full (e.g. "downstream").
                Defaults to "downstream".

        Returns:
            str: A `queue name=... leaky=... max-size-buffers=... ...`
                GStreamer pipeline fragment, terminated with " ! " so it can
                be concatenated directly into a larger pipeline description.
        '''
        return f"queue name={name} leaky={leaky} max-size-buffers={max_size_buffers} max-size-bytes=0 max-size-time=0 ! "

    pipeline_str = (
        # Camera source at native resolution
        "libcamerasrc name=libcamerasrc ! "
        "video/x-raw,width=1920,height=1080,framerate=30/1,format=NV12 ! "

        # Split into two paths
        "tee name=t allow-not-linked=true "  # CRITICAL: allow-not-linked prevents blocking

        # --- Path 1: AI Processing ---
        "t. ! " +
        QUEUE("queue_ai_scale", max_size_buffers=2, leaky="downstream") +
        "videoscale ! "
        "video/x-raw,width=640,height=640 ! "  # Scale to model input size
        "videoconvert ! "
        "video/x-raw,format=RGB ! "  # Convert to RGB for AI
        f"hailonet hef-path={model_path} is-active=true ! " +
        QUEUE("queue_hailofilter", max_size_buffers=2, leaky="downstream") +
        # Note: hailofilter thresholds will be applied in Python filtering below
        f"hailofilter so-path={postprocess_so} qos=false ! " +
        QUEUE("queue_appsink", max_size_buffers=2, leaky="downstream") +
        "appsink name=appsink emit-signals=true sync=false max-buffers=2 drop=true async=false " + 
        
        # --- Path 2: Network Stream ---
        "t. ! " +
        QUEUE("queue_network", max_size_buffers=5, leaky="downstream") +
        "videoscale ! video/x-raw,width=1280,height=720 ! "  # Reduce resolution for network
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency bitrate=3000 speed-preset=superfast key-int-max=15 ! "  # Force keyframe every 15 frames (0.5s)
        "h264parse ! "  # CRITICAL: Parse H264 stream before RTP payload
        "rtph264pay config-interval=10 pt=96 ! "  # Send SPS/PPS every 10 packets
        f"udpsink host={GROUND_STATION_IP} port={VIDEO_STREAM_PORT} sync=false async=false"
    )

    print(f"\nUsing pipeline:\n{pipeline_str}\n")
    
    # --- Start All Threads ---
    threads = [
        threading.Thread(name="gstreamer",    target=run_gstreamer,      args=(app_state, main_loop, pipeline_str), daemon=True),
        threading.Thread(name="command",      target=run_command_server, args=(app_state,),                         daemon=True),
        threading.Thread(name="drone_control",target=run_drone_control,  args=(app_state, drone_controller),        daemon=True),
    ]
    for t in threads:
        t.start()
    try:
        while True:
            if app_state.gst_error_event.is_set():
                print("Main thread: Error detected, exiting...")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutdown requested by user (Ctrl+C)...")
    finally:
        print("Shutting down...")
        app_state.gst_error_event.set()
        app_state.command_thread_stop_event.set()
        app_state.control_loop_stop_event.set()
        app_state.data_sender_stop_event.set()
        if main_loop.is_running():
            print("Quitting GLib main loop...")
            main_loop.quit()
        # TTL for thread joining to prevent hanging indefinitely
        print("Waiting for threads to join...")
        for t in threads:
            t.join(timeout=3)
        """        
        gst_thread.join(timeout=3)
        command_thread.join(timeout=2)
        control_thread.join(timeout=2)
        """

        # Close the UDP socket
        app_state.data_socket.close()

        # Final cleanup if necessary
        print("Drone tracker stopped")
        sys.exit(0)
        
if __name__ == "__main__":
    main()
