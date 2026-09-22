import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from pathlib import Path
import time
import socket
import struct
import math
import json
import os
import asyncio
import logging
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from pymavlink import mavutil
from common.sccs_common.config import load_config, get_setting


logging.basicConfig(filename = './app.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

_CFG = load_config()
_NET = _CFG.get("network", {})
GROUND_STATION_IP = get_setting(_NET, "ground_ip", "10.5.0.1")
VIDEO_STREAM_PORT = int(get_setting(_NET, "video_port", 5602))
DATA_PORT = int(get_setting(_NET, "data_port", 5601))
COMMAND_PORT = int(get_setting(_NET, "command_port", 5603))
# get_setting() (unlike a plain dict.get) falls back to the default below even
# when config.toml has drone_system_address explicitly set to "" -- see
# sccs_common/config.py's docstring for why the plain-.get() version of this
# line was a real bug in the original codebase.
DRONE_SYSTEM_ADDRESS = get_setting(_NET, "drone_system_address", "serial:///dev/ttyAMA0:57600")


# -----------------------------------------------------------------------------------------------
# GStreamer Thread
# -----------------------------------------------------------------------------------------------
def run_gstreamer(app_state, main_loop, pipeline_str):
    from hailo_inference import on_new_hailo_sample
    
    pipeline = None #default pipeline state to allow for graceful error handling
    try:
        pipeline = Gst.parse_launch(pipeline_str)
        
        appsink = pipeline.get_by_name("appsink") #parse appsink from pipeline
        if not appsink:
            print("ERROR: Could not find 'appsink' element in pipeline.")
            raise RuntimeError("Missing appsink element")

        appsink.set_property('emit-signals', True) #Enable signal emmisions for new samples
        appsink.set_property('max-buffers', 2)     #Small frame buffer to prevent latency build-up (set to 1 on ground)
        appsink.set_property('drop', True)         #Drop old frames to smoooth feed
        appsink.set_property('async', False)       #Make pull-sample blocking to sync with processing
        appsink.connect('new-sample', on_new_hailo_sample, app_state) # run new hailo sample

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            bus = pipeline.get_bus()
            err = bus.timed_pop_filtered(Gst.SECOND * 5, Gst.MessageType.ERROR)
            if err:
                err, debug = err.parse_error()
                print(f"GStreamer ERROR on startup: {err} | Debug: {debug}")
            else:
                print("ERROR: Could not set pipeline to PLAYING state.")
            raise RuntimeError("Pipeline failed to start")
        elif ret == Gst.StateChangeReturn.NO_PREROLL:
            print("Pipeline is live and does not need preroll.")
        
        print("GStreamer pipeline is running...")

        main_loop.run()

    except GLib.Error as e:
        print(f"GStreamer GLib.Error: {e}")
    except Exception as e:
        print(f"GStreamer error: {e}")
    finally:
        print("GStreamer thread stopping...")
        if pipeline:
            pipeline.set_state(Gst.State.NULL)
        app_state.gst_error_event.set()

# -----------------------------------------------------------------------------------------------
# Telemetry Sender
# -----------------------------------------------------------------------------------------------



from sccs_common.protocol import PACKET_HEADER_FMT, PACKET_HEADER_LEN, PACKET_MAX_PAYLOAD, pack_fragments

def send_telemetry_udp(sock, addr, seq, data_bytes):

    # Fragmentation itself now lives in sccs_common.protocol.pack_fragments,
    # shared with (and unit-tested via) common/tests/test_protocol.py, so the
    # exact framing logic used here is the same logic the tests exercise.

    # Send each fragment and handle errors per-fragment. Return True only if all fragments were sent.
    
    try:
        for frag_id, packet in enumerate(pack_fragments(seq, data_bytes)):
            try:
                sock.sendto(packet, addr)
            except socket.error as e:
                print(f"send_telemetry_udp: sendto failed for frag {frag_id}: {e}", end='\r')
                return False
    except Exception as e:
        print(f"send_telemetry_udp: error preparing fragments: {e}")
        return False

    return True

def run_command_server(app_state):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # Enable keepalive
        s.bind(('', COMMAND_PORT))
        s.listen(5)
        s.settimeout(1.0)
        logging.info(f"Command server listening on port {COMMAND_PORT}")

        while not app_state.command_thread_stop_event.is_set():
            try:
                conn, addr = s.accept()
                conn.settimeout(3.0)  # 3-second timeout for receive
                with conn:
                    data = conn.recv(1024)
                    if data:
                        logging.info(f"Received command from {addr}: {data.decode('utf-8')}")
                        command = json.loads(data.decode('utf-8'))
                        if 'target_id' in command:
                            new_target_id = command['target_id']
                            with app_state.target_lock:
                                app_state.target_id = new_target_id

                            with app_state.drone_state_lock:
                                if new_target_id is not None:
                                    if app_state.drone_state == "MANUAL":
                                        app_state.drone_state = "TRACKING"
                                        logging.info(f"\n[MANUAL] → TRACKING: Target ID {new_target_id} selected.")
                                    elif app_state.drone_state == "TRACKING":
                                        print(f"\n[TRACKING] Target switched to ID {new_target_id}.")
                                else:
                                    app_state.drone_state = "MANUAL"
                                    logging.info(f"\nTarget cleared by operator. Returning to MANUAL.")
                    else:
                        logging.info("Received empty command.")
            except socket.timeout:
                continue  # Normal timeout, loop again
            except json.JSONDecodeError:
                logging.warning("Received invalid command (not JSON)")
            except Exception as e:
                if not app_state.command_thread_stop_event.is_set():
                    logging.error(f"Command server error: {e}")
                    
# -----------------------------------------------------------------------------------------------
# Drone Control Loop (MAVLink) Thread
# -----------------------------------------------------------------------------------------------

TARGET_LOST_HOVER_TIMEOUT = 5.0
HOLD_BEFORE_MANUAL_DURATION = 2.0
STATE_CYCLE_INTERVAL = 0.1

def run_drone_control(app_state, drone_controller):
    asyncio.run(run_drone_control_async(app_state, drone_controller))


async def run_drone_control_async(app_state, drone_controller, bbox=None):
        
    drone = System()
    logging.info(f"Connecting to drone via MAVLink on {DRONE_SYSTEM_ADDRESS}...")
 
    try:
        await drone.connect(system_address=DRONE_SYSTEM_ADDRESS)
 
        async for state in drone.core.connection_state():
            if state.is_connected:
                logging.info("Drone connected!")
                break
 
        logging.info("Waiting for drone to be ready...")
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                logging.info("Drone is ready.")
                break
 
        logging.info("Starting in MANUAL mode. Click a target to engage autonomous tracking.")
 
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        offboard_active = False
 
        hover_lost_time   = None   
        returning_start   = None   
 
        last_command_str  = None
 
        while not app_state.control_loop_stop_event.is_set():
            try:
                with app_state.drone_state_lock:
                    current_state = app_state.drone_state
 
                with app_state.target_lock:
                    target_id = app_state.target_id
 
                with app_state.frame_size_lock:
                    frame_w = app_state.frame_width
                    frame_h = app_state.frame_height
 
        
                target_bbox = None
                if target_id is not None:
                    with app_state.tracker_lock:
                        target_data = app_state.tracker.tracked_objects.get(target_id)
                        if target_data:
                            target_bbox = target_data.get('bbox')
 
           
                target_visible = (target_id is not None) and (target_bbox is not None)
 
        
                if current_state == "MANUAL":
                    if offboard_active:
                        try:
                            await drone.offboard.stop()
                            offboard_active = False
                            logging.info("Offboard stopped — MANUAL mode.")
                        except OffboardError as e:
                            logging.error(f"Failed to stop offboard: {e}")
 
            
                    hover_lost_time = None
                    returning_start = None
 
                    await asyncio.sleep(STATE_CYCLE_INTERVAL)
                    continue
 
        
                elif current_state == "TRACKING":
                    if not offboard_active:
                        try:
                            await drone.offboard.start()
                            offboard_active = True
                            logging.info("Offboard started — TRACKING mode.")
                        except OffboardError as e:
                            logging.error(f"Failed to start offboard: {e}")
                            await asyncio.sleep(STATE_CYCLE_INTERVAL)
                            continue
 
                 
                    hover_lost_time = None
 
                    if target_visible:
                        forward_velocity, up_velocity, yaw_velocity, command_str = \
                            drone_controller.compute_command(target_bbox, frame_w, frame_h)
 
                        with app_state.velocity_lock:
                            app_state.forward_velocity = forward_velocity
                            app_state.up_velocity      = up_velocity
                            app_state.yaw_velocity     = yaw_velocity
 
                        await drone.offboard.set_velocity_body(
                            VelocityBodyYawspeed(forward_velocity, 0.0, -up_velocity, yaw_velocity)
                        )
 
                        if command_str != last_command_str:
                            logging.info(command_str)
                            last_command_str = command_str
 
                    else:
                        logging.info("[TRACKING] Target lost by tracker. → HOVERING.")
                        hover_lost_time = time.monotonic()
                        with app_state.drone_state_lock:
                            app_state.drone_state = "HOVERING"
               
                elif current_state == "HOVERING":
                    if not offboard_active:
                        try:
                            await drone.offboard.start()
                            offboard_active = True
                            logging.info("Offboard restarted for HOVERING hold.")
                        except OffboardError as e:
                            logging.error(f"Failed to start offboard during HOVERING: {e}")
                            await asyncio.sleep(STATE_CYCLE_INTERVAL)
                            continue
 

                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                    )
 
                    if target_visible:
                        logging.info("[HOVERING] Target reacquired. → TRACKING.")
                        hover_lost_time = None
                        with app_state.drone_state_lock:
                            app_state.drone_state = "TRACKING"
 
                    else:
                        elapsed = time.monotonic() - hover_lost_time
                        remaining = TARGET_LOST_HOVER_TIMEOUT - elapsed
 
                        if elapsed >= TARGET_LOST_HOVER_TIMEOUT:
                            logging.info(
                                f"[HOVERING] Timeout ({TARGET_LOST_HOVER_TIMEOUT}s) elapsed. → RETURNING."
                            )
                            returning_start = time.monotonic()
                            with app_state.drone_state_lock:
                                app_state.drone_state = "RETURNING"
                        else:
                            logging.debug(f"[HOVERING] Holding... ({remaining:.1f}s remaining)")
 
                elif current_state == "RETURNING":
                    if not offboard_active:
                        try:
                            await drone.offboard.start()
                            offboard_active = True
                            logging.info("Offboard restarted for RETURNING hold.")
                        except OffboardError as e:
                            logging.error(f"Failed to start offboard during RETURNING: {e}")
                            await asyncio.sleep(STATE_CYCLE_INTERVAL)
                            continue
 
                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                    )
 
                    elapsed = time.monotonic() - returning_start
                    remaining = HOLD_BEFORE_MANUAL_DURATION - elapsed
 
                    if elapsed >= HOLD_BEFORE_MANUAL_DURATION:
                        logging.info("[RETURNING] Hold complete. Stopping offboard. → MANUAL.")
                        try:
                            await drone.offboard.stop()
                            offboard_active = False
                        except OffboardError as e:
                            logging.error(f"Failed to stop offboard during RETURNING: {e}")
 
                        with app_state.drone_state_lock:
                            app_state.drone_state = "MANUAL"
                        with app_state.target_lock:
                            app_state.target_id = None
 
                        returning_start = None
                        hover_lost_time = None
                    else:
                        logging.debug(f"[RETURNING] Holding before handoff... ({remaining:.1f}s remaining)")
 
                else:
                    logging.warning(f"Unknown drone state: '{current_state}'. Treating as MANUAL.")
                    with app_state.drone_state_lock:
                        app_state.drone_state = "MANUAL"
 
                await asyncio.sleep(STATE_CYCLE_INTERVAL)
 
            except Exception as e:
                if not app_state.control_loop_stop_event.is_set():
                    logging.error(f"Error in control loop: {e}")
                    await asyncio.sleep(STATE_CYCLE_INTERVAL)
 
    except Exception as e:
        logging.error(f"Drone connection/setup error: {e}")
 
    finally:
        logging.info("Drone control loop shutting down...")
        if offboard_active:
            try:
                await drone.offboard.stop()
            except Exception:
                pass
        try:
            await drone.disconnect()
        except Exception as e:
            logging.error(f"Error during drone disconnect: {e}")
 
