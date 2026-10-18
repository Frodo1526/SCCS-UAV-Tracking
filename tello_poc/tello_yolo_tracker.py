from djitellopy import Tello
import cv2
import time
import threading
import queue
import pygame
import numpy as np
from ultralytics import YOLO
import torch

# --- Parameters & Optimizations ---
S = 60
RC_CONTROL_RATE = 20
CONFIDENCE_THRESHOLD = 0.4
YAW_SMOOTHING = 0.90
PITCH_SMOOTHING = 0.95
FORWARD_SMOOTHING = 0.95
TARGET_BOX_HEIGHT_RATIO = 0.65
TRACKER_CONFIG = 'bytetrack.yaml'
DISPLAY_FPS = 60

# --- PERFORMANCE OPTIMIZATIONS ---
YOLO_INPUT_WIDTH = 640
YOLO_INPUT_HEIGHT = 480
# The Tello drone streams at 30 FPS. We will pace our processing loop to match this.
TARGET_FPS = 30.0

class TelloYoloTracker:
    """
    A controller for real-time YOLO tracking with a Tello drone,
    with frame pacing and color correction for a smooth, tear-free video feed.
    """
    def __init__(self):
        self.tello = Tello()
        self.running = True
        self.send_rc_control = False
        self.program_state = "MANUAL_FLIGHT"
        self.input_id_str = ""

        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.auto_yaw_velocity = 0
        self.auto_pitch_velocity = 0
        self.auto_forward_velocity = 0
        self.tracked_id = None
        self.trigger_up_velocity = 0
        self.trigger_down_velocity = 0
        
        self.processed_frame_queue = queue.LifoQueue(maxsize=1)
        self.command_queue = queue.Queue()

        pygame.init()
        pygame.display.set_caption("Tello YOLO Tracker")
        self.screen = pygame.display.set_mode((960, 720), pygame.DOUBLEBUF | pygame.HWSURFACE)
        self.font = pygame.font.Font(None, 36)

        self.joystick = None
        self.joystick_deadzone = 0.15
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"✅ Controller connected: {self.joystick.get_name()}")
        else:
            print("⚠️ No controller detected. Using keyboard controls.")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ Using device: {self.device.upper()}")
        print("Loading YOLO model...")
        self.model = YOLO('yolo12n.pt')
        print("✅ Model loaded successfully.")

        self.frame_read = None

    def run(self):
        """Main loop: connect, start threads, handle UI."""
        try:
            self.tello.connect()
            self.tello.set_speed(S)
            self.tello.streamoff()
            self.tello.streamon()
            print(f"✅ Battery: {self.tello.get_battery()}%")

            self.frame_read = self.tello.get_frame_read()
            time.sleep(1)

            threads = [
                threading.Thread(target=self._yolo_processing_thread, daemon=True),
                threading.Thread(target=self._command_thread, daemon=True),
                threading.Thread(target=self._rc_control_thread, daemon=True)
            ]
            for t in threads: t.start()

            clock = pygame.time.Clock()
            while self.running:
                self.handle_pygame_events()
                try:
                    processed_frame = self.processed_frame_queue.get(block=False)
                    self.display_frame(processed_frame)
                except queue.Empty:
                    pass
                clock.tick(DISPLAY_FPS)

        except Exception as e:
            print(f"❌ An error occurred in the main loop: {e}")
        finally:
            self.cleanup()

    def _yolo_processing_thread(self):
        """
        Grabs, processes, and paces frames to ensure a smooth, real-time video feed.
        """
        target_frame_duration = 1.0 / TARGET_FPS

        while self.running:
            start_time = time.time()
            try:
                original_frame_bgr = self.frame_read.frame
                if original_frame_bgr is None:
                    continue

                original_frame_rgb = cv2.cvtColor(original_frame_bgr, cv2.COLOR_BGR2RGB)
                yolo_input_frame = cv2.resize(original_frame_rgb, (YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT))
                
                results = self.model.track(
                    yolo_input_frame,
                    persist=True, conf=CONFIDENCE_THRESHOLD, device=self.device,
                    verbose=False, classes=[0, 2, 3, 7], tracker=TRACKER_CONFIG, half=True
                )

                # --- BOUNDING BOX FIX ---
                # 1. Plot results on the same small frame used for inference.
                #    This ensures boxes are drawn correctly relative to the image content.
                if results[0].boxes is not None:
                    # .plot() returns a BGR frame, convert it to RGB for consistency
                    annotated_small_frame_bgr = results[0].plot(conf=False)
                    annotated_small_frame = cv2.cvtColor(annotated_small_frame_bgr, cv2.COLOR_BGR2RGB)
                else:
                    # yolo_input_frame is already in RGB
                    annotated_small_frame = yolo_input_frame

                # 2. Resize the fully annotated small frame back up to the original display size.
                annotated_frame = cv2.resize(annotated_small_frame, (original_frame_bgr.shape[1], original_frame_bgr.shape[0]))
                
                target_found = self.update_auto_tracking_velocities(results, original_frame_bgr.shape[1], original_frame_bgr.shape[0])

                if self.program_state == "TRACKING" and target_found and results[0].boxes is not None and results[0].boxes.id is not None:
                    for box in results[0].boxes:
                        if box.id is not None and int(box.id[0]) == self.tracked_id:
                            # --- RED OVERLAY ---
                            # Use the scaled coordinates to draw a semi-transparent overlay.
                            xmin, ymin, xmax, ymax = self.scale_bbox(box.xyxy[0], original_frame_bgr.shape)
                            overlay = annotated_frame.copy()
                            # Draw a filled red rectangle on the copy
                            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (255, 0, 0), -1) 
                            alpha = 0.4  # Transparency factor.
                            # Blend the overlay with the original frame
                            annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)
                            break
                
                if not self.processed_frame_queue.full():
                    self.processed_frame_queue.put(annotated_frame)

            except Exception as e:
                print(f"Error in YOLO processing thread: {e}")
                time.sleep(0.5)

            elapsed_time = time.time() - start_time
            wait_time = target_frame_duration - elapsed_time
            if wait_time > 0:
                time.sleep(wait_time)

    def scale_bbox(self, bbox_tensor, original_frame_shape):
        """Scales bounding box coordinates from YOLO input size to original frame size."""
        xmin_yolo, ymin_yolo, xmax_yolo, ymax_yolo = bbox_tensor.cpu().numpy()
        orig_h, orig_w, _ = original_frame_shape
        scale_x, scale_y = orig_w / YOLO_INPUT_WIDTH, orig_h / YOLO_INPUT_HEIGHT
        return int(xmin_yolo * scale_x), int(ymin_yolo * scale_y), int(xmax_yolo * scale_x), int(ymax_yolo * scale_y)

    def display_frame(self, frame_rgb):
        """Displays the frame, now expecting it to be in RGB format."""
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        self.screen.blit(pygame.transform.scale(frame_surface, self.screen.get_size()), (0, 0))

        screen_center_x, screen_center_y = self.screen.get_width() // 2, self.screen.get_height() // 2
        pygame.draw.line(self.screen, (255, 255, 255), (screen_center_x - 15, screen_center_y), (screen_center_x + 15, screen_center_y), 2)
        pygame.draw.line(self.screen, (255, 255, 255), (screen_center_x, screen_center_y - 15), (screen_center_x, screen_center_y + 15), 2)

        try:
            battery_level = self.tello.get_battery()
            color = (0, 255, 0) if battery_level > 20 else (255, 0, 0)
            text_surface = self.font.render(f"Battery: {battery_level}%", True, color)
            bg_rect = pygame.Rect(self.screen.get_width() - text_surface.get_width() - 20, 10, text_surface.get_width() + 10, text_surface.get_height() + 10)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA); bg_surface.fill((0, 0, 0, 120))
            self.screen.blit(bg_surface, bg_rect.topleft)
            self.screen.blit(text_surface, (bg_rect.x + 5, bg_rect.y + 5))
        except: pass

        if self.program_state == "AWAITING_ID":
            prompt_text = f"Enter ID to Track: {self.input_id_str}"
            text_surface = self.font.render(prompt_text, True, (255, 255, 255))
            bg_rect = pygame.Rect(10, 10, text_surface.get_width() + 20, text_surface.get_height() + 20)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA); bg_surface.fill((0, 0, 0, 120))
            self.screen.blit(bg_surface, bg_rect.topleft)
            self.screen.blit(text_surface, (20, 20))

        pygame.display.flip()
        
    def update_auto_tracking_velocities(self, results, frame_width, frame_height):
        frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
        target_yaw_velocity, target_pitch_velocity, target_forward_velocity = 0, 0, 0
        id_found = False

        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box in results[0].boxes:
                if box.id is not None and int(box.id[0]) == self.tracked_id:
                    id_found = True
                    xmin, ymin, xmax, ymax = self.scale_bbox(box.xyxy[0], (frame_height, frame_width, 3))
                    
                    relative_x = ((xmin + xmax) / 2) - frame_center_x
                    relative_y = frame_center_y - ((ymin + ymax) / 2)
                    height_error = (frame_height * TARGET_BOX_HEIGHT_RATIO) - (ymax - ymin)
                    
                    if abs(relative_x) > 50: target_yaw_velocity = int(np.clip(relative_x * 0.2, -S, S))
                    if abs(relative_y) > 35: target_pitch_velocity = int(np.clip(relative_y * 0.3, -S, S))
                    if abs(height_error) > frame_height * 0.05: target_forward_velocity = int(np.clip(height_error * 0.4, -S, S))
                    break
        
        self.auto_yaw_velocity = int(self.auto_yaw_velocity * YAW_SMOOTHING + target_yaw_velocity * (1 - YAW_SMOOTHING))
        self.auto_pitch_velocity = int(self.auto_pitch_velocity * PITCH_SMOOTHING + target_pitch_velocity * (1 - PITCH_SMOOTHING))
        self.auto_forward_velocity = int(self.auto_forward_velocity * FORWARD_SMOOTHING + target_forward_velocity * (1 - FORWARD_SMOOTHING))
        return id_found

    def _rc_control_thread(self):
        while self.running:
            try:
                if self.send_rc_control:
                    fb = self.for_back_velocity if self.for_back_velocity != 0 or self.program_state != "TRACKING" else self.auto_forward_velocity
                    yaw = self.yaw_velocity if self.yaw_velocity != 0 or self.program_state != "TRACKING" else self.auto_yaw_velocity
                    ud = self.up_down_velocity if self.up_down_velocity != 0 or self.program_state != "TRACKING" else self.auto_pitch_velocity
                    self.tello.send_rc_control(self.left_right_velocity, fb, ud, yaw)
                time.sleep(1 / RC_CONTROL_RATE)
            except Exception as e: print(f"Error in RC control thread: {e}"); break
            
    def handle_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            elif event.type == pygame.KEYDOWN: self.handle_key_down(event.key)
            elif event.type == pygame.KEYUP: self.handle_key_up(event.key)
            elif event.type == pygame.JOYAXISMOTION: self.handle_joystick_axis(event)
            elif event.type == pygame.JOYBUTTONDOWN: self.handle_joystick_button_down(event)
            elif event.type == pygame.JOYBUTTONUP: self.handle_joystick_button_up(event)

    def handle_joystick_axis(self, event):
        if self.joystick is None or self.program_state == "AWAITING_ID": return
        if event.axis == 0: self.left_right_velocity = int(event.value * S) if abs(event.value) > self.joystick_deadzone else 0
        elif event.axis == 1: self.for_back_velocity = int(-event.value * S) if abs(event.value) > self.joystick_deadzone else 0
        elif event.axis == 4: self.trigger_up_velocity = int(((event.value + 1) / 2) * S)
        elif event.axis == 5: self.trigger_down_velocity = int(((event.value + 1) / 2) * S)
        self.up_down_velocity = self.trigger_up_velocity - self.trigger_down_velocity

    def handle_joystick_button_down(self, event):
        if self.joystick is None or self.program_state == "AWAITING_ID": return
        if event.button == 0: self.command_queue.put(self._takeoff_and_enable_rc)
        elif event.button == 1: self.command_queue.put(lambda: self.tello.land())
        elif event.button == 4: self.yaw_velocity = -S
        elif event.button == 5: self.yaw_velocity = S
        elif event.button == 6: self.cancel_tracking()
        elif event.button == 7: self.program_state = "AWAITING_ID"; self.input_id_str = ""
    
    def handle_joystick_button_up(self, event):
        if self.joystick and (event.button == 4 or event.button == 5): self.yaw_velocity = 0

    def handle_key_down(self, key):
        if self.program_state == "AWAITING_ID": self.handle_id_input(key); return
        if key == pygame.K_g: self.command_queue.put(lambda: self.tello.land())
        elif key == pygame.K_t: self.command_queue.put(self._takeoff_and_enable_rc)
        elif key == pygame.K_ESCAPE: self.cancel_tracking()
        elif key == pygame.K_BACKQUOTE: self.program_state = "AWAITING_ID"; self.input_id_str = ""
        elif key == pygame.K_w: self.for_back_velocity = S
        elif key == pygame.K_s: self.for_back_velocity = -S
        elif key == pygame.K_a: self.left_right_velocity = -S
        elif key == pygame.K_d: self.left_right_velocity = S
        elif key == pygame.K_UP: self.up_down_velocity = S
        elif key == pygame.K_DOWN: self.up_down_velocity = -S
        elif key == pygame.K_q or key == pygame.K_LEFT: self.yaw_velocity = -S
        elif key == pygame.K_e or key == pygame.K_RIGHT: self.yaw_velocity = S

    def handle_id_input(self, key):
        if key == pygame.K_RETURN and self.input_id_str:
            try:
                self.tracked_id = int(self.input_id_str)
                print(f"Target ID set to {self.tracked_id}.")
                self.command_queue.put(self._initiate_tracking_sequence)
                self.program_state = "TRACKING"
            except ValueError: print("Invalid ID.")
            self.input_id_str = ""
        elif key == pygame.K_BACKSPACE: self.input_id_str = self.input_id_str[:-1]
        elif pygame.K_0 <= key <= pygame.K_9: self.input_id_str += chr(key)
        elif key == pygame.K_ESCAPE: self.program_state = "MANUAL_FLIGHT"; self.input_id_str = ""

    def handle_key_up(self, key):
        if key in [pygame.K_w, pygame.K_s]: self.for_back_velocity = 0
        elif key in [pygame.K_a, pygame.K_d]: self.left_right_velocity = 0
        elif key in [pygame.K_UP, pygame.K_DOWN]: self.up_down_velocity = 0
        elif key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_q, pygame.K_e]: self.yaw_velocity = 0

    def cancel_tracking(self):
        if self.program_state == "TRACKING":
            print("Tracking cancelled. Returning to manual control.")
            self.program_state = "MANUAL_FLIGHT"
            self.tracked_id = None
            self.auto_yaw_velocity, self.auto_pitch_velocity, self.auto_forward_velocity = 0, 0, 0

    def _command_thread(self):
        while self.running:
            try:
                command = self.command_queue.get(block=True, timeout=1.0)
                command()
            except queue.Empty: continue
    
    def _takeoff_and_enable_rc(self):
        if not self.tello.is_flying: self.tello.takeoff()
        self.send_rc_control = True

    def _initiate_tracking_sequence(self):
        try:
            if not self.tello.is_flying:
                print("🚀 Target selected. Taking off to begin tracking...")
                self.tello.takeoff()
                time.sleep(2.0)
            self.send_rc_control = True
            print("✅ Auto-tracking enabled.")
        except Exception as e: print(f"❌ Takeoff failed during tracking initiation: {e}")

    def cleanup(self):
        print("🧹 Starting cleanup...")
        self.running = False
        time.sleep(1.5)
        if self.tello.is_flying: self.tello.land()
        self.tello.streamoff()
        self.tello.end()
        pygame.quit()
        print("✅ Shutdown complete.")

if __name__ == '__main__':
    tracker = TelloYoloTracker()
    tracker.run()

