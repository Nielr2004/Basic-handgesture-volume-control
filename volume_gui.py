import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import numpy as np
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Volume control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# MediaPipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

def calculate_angle(p1, p2, p3):
    """Returns angle between points (index, thumb, wrist) in degrees."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def start_twist_volume_control():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        lmList = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lm = handLms.landmark
                h, w, _ = img.shape
                # Get index (8), thumb (4), and wrist (0)
                p1 = (int(lm[8].x * w), int(lm[8].y * h))   # Index
                p2 = (int(lm[4].x * w), int(lm[4].y * h))   # Thumb
                p3 = (int(lm[0].x * w), int(lm[0].y * h))   # Wrist

                angle = calculate_angle(p1, p2, p3)
                angle = np.clip(angle, 20, 160)  # Angle between thumb and index

                # Map angle to volume range
                vol = np.interp(angle, [20, 160], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

                # Volume bar
                volPerc = np.interp(angle, [20, 160], [0, 100])
                volBar = np.interp(angle, [20, 160], [400, 150])

                # Draw
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                cv2.putText(img, f'{int(volPerc)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

        cv2.imshow("Twist Gesture Volume Control", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI setup
def launch_gui():
    root = tk.Tk()
    root.title("Twist Volume Controller")
    root.geometry("400x200")

    tk.Label(root, text="Twist your hand (thumb + index) to control volume", font=("Arial", 12)).pack(pady=20)

    def on_start():
        threading.Thread(target=start_twist_volume_control, daemon=True).start()

    tk.Button(root, text="Start Twist Control", font=("Arial", 12), command=on_start).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    launch_gui()
