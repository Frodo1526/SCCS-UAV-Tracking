from djitellopy import Tello
import threading
import cv2
import time
import sys
from ultralytics import YOLO

import logging
import signal

import app_state
import ui
from app_state import AppState
from ui import make_mouse_callback, draw_overlay
from yolo_inference import run_inference_loop
from ctrl import run_control_loop
from pynput import keyboard

logging.basicConfig(filename = './app.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Video frame config:
FRAME_WIDTH = 960 # Tello default camera width
FRAME_HEIGHT = 720 # Tello default camera height
FPS_TARGET = 30

# YOLO model
YOLO_MODEL = 'yolov8s.pt'

# OpenCV window name
WINDOW_NAME = 'Tello POC'



def battery_refresh_loop(tello, app_state):
    """Periodically refresh battery status on a separate thread to avoid blocking display."""
    battery = 100
    while not app_state.stop_event.is_set():
        try:
            time.sleep(30)  # Refresh every 30 seconds
            battery = tello.get_battery()
            with app_state.battery_lock:
                app_state.battery = battery
            logging.info(f"Battery: {battery}%")
        except Exception as e:
            logging.warning(f"Couldn't refresh battery: {e}")
    
def start_tello(detector, settings):
    logging.info(f"Using detector: {detector.description}")

    logging.info("Connecting to Tello...")
    tello = Tello()
    tello.connect()
    battery = tello.get_battery()
    logging.info(f"Connected, Battery: {battery}%")
    
    if battery < 20:
        logging.warning(f"Low battery!")
        
    tello.streamon()
    
    time.sleep(2)  # give stream time to initialize
    
    frame_read = tello.get_frame_read()
    
    logging.info("Waiting for first frame...")
    timeout = time.time() + 10.0
    while frame_read.frame is None:
        if time.time() > timeout:
            logging.error("Timed out waiting for first frame. Check Tello WiFi connection.")
            tello.streamoff()
            tello.end()
            sys.exit(1)
        time.sleep(0.1)
    logging.info("First frame received.")
    
    app_state = AppState()
    app_state.battery = battery
    
    app_state.airborne = False
    
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(app_state))



    # Start battery refresh thread (non-blocking)
    battery_thread = threading.Thread(
        target=battery_refresh_loop, args=(tello, app_state), daemon=True
    )
    battery_thread.start()

    inference_thread = threading.Thread(
        target=run_inference_loop,
        args=(
            detector,
            frame_read,
            app_state,
            FRAME_WIDTH,
            FRAME_HEIGHT
        ),
        daemon=True
    )


    inference_thread.start()

    app_state.frame = tello.get_frame_read().frame

    while app_state.frame is None:

        time.sleep(0.1)

    # bfrost 9/29
    ui_thread = threading.Thread(
        target=ui.draw_overlay, args=(tello, app_state,), daemon=True
    )
    ui_thread.start()

    # bfrost
    on_press, on_release = make_keyboard_callbacks(app_state)

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )

    listener.start()

    logging.info(f"Running. T=takeoff, L=land, C=clear target, Q=Quit. Click box to track. WASD=move, RF=up/down")

    takeoff_thread = None
    land_thread = None
    control_thread = None
    
    try:
        while True:


            with app_state.state_lock:
                state = app_state.drone_state
                
            with app_state.target_lock:
                current_target = app_state.target_id

            with app_state.battery_lock:
                battery = app_state.battery
            
            with app_state.command_lock:
                cmd_str = app_state.current_command_str
            
            if time.time() - app_state.start_time % 10 < 0.1:  # Log status every 10 seconds
                logging.info(f"[10s Check] State: {state}, Target: {current_target}, Battery: {battery}%, Command: {cmd_str}")

            with app_state.display_lock: # Bfrost 9.29
                display = app_state.display # Bfrost 9.29

            if display is None:
                logging.warning("Overlay skipped due to error.")
                display = app_state.frame.copy()

            if time.time() - app_state.last_frame_time > 3.0:
                logging.warning("Video stream stalled...")


            cv2.imshow(WINDOW_NAME, display)
            cv2.waitKey(1)

            key = app_state.pressed_keys

            if 't' in key or 'T' in key: #bfrost
                if not app_state.airborne:
                    def do_takeoff():

                        try:
                            nonlocal control_thread
                            with app_state.keys_lock:
                                app_state.pressed_keys.discard('t')
                                app_state.pressed_keys.discard('T')

                            if app_state.control_stop.is_set():
                                app_state.control_stop.clear()  # Ensure control loop is running
                            logging.info("Takeoff.")
                            tello.takeoff()
                            # Start a new control thread for this flight session
                            control_thread = threading.Thread(
                                target=run_control_loop, args=(tello, app_state), daemon=True
                            )
                            control_thread.start()
                            app_state.airborne = True
                            logging.info("Airborne.")
                        except Exception as e:
                            logging.error(f"Takeoff error: {e}")
                    takeoff_thread = threading.Thread(target=do_takeoff, daemon=True)
                    takeoff_thread.start()

            elif 'l' in key or 'L' in key: #bfrost
                def land_tello():
                    try:
                        with app_state.keys_lock:
                            app_state.pressed_keys.discard('l')
                            app_state.pressed_keys.discard('L')

                        app_state.control_stop.set()  # Signal control loop to stop sending commands
                        # Stop the control loop from sending tracking commands
                        with app_state.state_lock:
                            app_state.drone_state = "MANUAL"
                        with app_state.target_lock:
                            app_state.target_id = None
                        # Give control loop time to stop sending RC commands
                        time.sleep(0.2)
                        logging.info("Landing.")


                        tello.land()
                        app_state.airborne = False
                    except Exception as e:
                        logging.error(f"Landing error (171): {e}")
                land_thread = threading.Thread(target=land_tello, daemon=True)
                land_thread.start()

            elif 'c' in key or 'C' in key: #bfrost
                with app_state.target_lock:
                    app_state.target_id = None
                with app_state.state_lock:
                    app_state.drone_state = "MANUAL"
                logging.info("Target cleared.")

            elif 'q' in key or 'Q' in key: #bfrost
                logging.info("Quitting...")
                break

            # Track pressed keys for manual control with age-based removal
            # Keys persist for 6 frames (200ms) after release to prevent jittery stopping
            with app_state.keys_lock:
                VALID_KEYS = set("WwSsAaDdRrFf")
                KEY_RELEASE_THRESHOLD = 6  # Frames before removing a released key

                # Add newly pressed key to the set and reset its age
                # Remove conflicting keys (e.g., if W pressed, remove S)
                if 'w' in key or 'W' in key:
                    app_state.pressed_keys.discard(ord('S'))
                    app_state.pressed_keys.discard(ord('s'))
                elif 's' in key or 'S' in key:
                    app_state.pressed_keys.discard(ord('W'))
                    app_state.pressed_keys.discard(ord('w'))
                elif 'a' in key or 'A' in key:
                    app_state.pressed_keys.discard(ord('D'))
                    app_state.pressed_keys.discard(ord('d'))
                elif 'd' in key or 'D' in key:
                    app_state.pressed_keys.discard(ord('A'))
                    app_state.pressed_keys.discard(ord('a'))
                elif 'r' in key or 'R' in key:
                    app_state.pressed_keys.discard(ord('F'))
                    app_state.pressed_keys.discard(ord('f'))
                elif 'f' in key or 'F' in key:
                    app_state.pressed_keys.discard(ord('R'))
                    app_state.pressed_keys.discard(ord('r'))
            
            
    except KeyboardInterrupt:
        logging.info("Interrupted.")
        
    finally:
        try:
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.2)
        except Exception:
            pass
        
        app_state.stop_event.set()
        logging.info("Stopping main loop...")
        
        # Join takeoff thread if it exists and is alive
        if takeoff_thread is not None and takeoff_thread.is_alive():
            takeoff_thread.join(timeout=2)
        
        # Join land thread if it exists and is alive (priority)
        if land_thread is not None and land_thread.is_alive():
            land_thread.join(timeout=5)
        
        # Join control thread if it exists and is alive
        if control_thread is not None and control_thread.is_alive():
            control_thread.join(timeout=2)

        if inference_thread.is_alive():
            inference_thread.join(timeout=2)

        if app_state.airborne:
            logging.info("Landing before exit...")
            original = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                tello.land()
                logging.info("Land command acknowledged.")
            except Exception as e:
                logging.error(f"Land error: {e}")
            finally:
                signal.signal(signal.SIGINT, original)
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()
        logging.info("Done.")
        sys.exit(0)


# bfrost
def make_keyboard_callbacks(app_state):

    def on_press(key):
        try:
            with app_state.keys_lock:
                app_state.pressed_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(key):
        try:
            with app_state.keys_lock:
                app_state.pressed_keys.discard(key.char)
        except AttributeError:
            pass

    return on_press, on_release

if __name__ == "__main__":
    start_tello()