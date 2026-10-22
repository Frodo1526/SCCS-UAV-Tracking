import cv2
import numpy as np
from pynput import keyboard

pressed = set()

def on_press(key):
    try:
        pressed.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        pressed.discard(key.char)
        print(f"Released: {key.char}")
    except AttributeError:
        pass

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)

listener.start()

img = np.full((500, 500, 3), 255, dtype=np.uint8)

cv2.imshow("White Image", img)
while True:
   print(pressed)
cv2.destroyAllWindows()