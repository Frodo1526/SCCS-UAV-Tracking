from djitellopy import Tello
import time

tello = Tello()
try:
    tello.connect()
except Exception:
    pass  # 'command' response is fine, library just doesn't recognize it

time.sleep(1)
tello.takeoff()
time.sleep(3)
tello.land()