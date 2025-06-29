import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import numpy as np
import mediapipe as mp
import math
import queue
import time
from pynput.keyboard import Controller, Key

# CustomTkinter for a modern UI look
import customtkinter as ctk

# PyCaw for system volume control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --- Global Variables and Setup ---
# Initialize keyboard controller for media keys
keyboard = Controller()

# Volume control setup
min_system_vol, max_system_vol = -65.25, 0.0 # Default range for pycaw
system_volume_interface = None
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    system_volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = system_volume_interface.GetVolumeRange()
    min_system_vol, max_system_vol = volRange[0], volRange[1]
except Exception as e:
    messagebox.showerror("Audio Device Error", f"Could not access audio device: {e}\n"
                                             "Volume control by gesture will be disabled.")

# MediaPipe setup with improved confidence and model complexity
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,         # Use the more accurate model (0 or 1, 1 is default for Hands)
    min_detection_confidence=0.8, # Increased for better initial detection
    min_tracking_confidence=0.7   # Increased for more robust tracking
)
mp_draw = mp.solutions.drawing_utils

# Threading flags and queues
running_gesture_control = True # Starts automatically
cap = None
ui_update_queue = queue.Queue()

# Gesture state variables to prevent rapid triggering
last_play_pause_time = 0
last_next_track_time = 0
last_mute_unmute_time = 0
GESTURE_COOLDOWN = 1.0 # seconds between gesture activations for media actions

# Volume freeze mechanism
volume_freeze_active = False
volume_freeze_start_time = 0
VOLUME_FREEZE_DURATION = 1.5 # seconds to freeze volume after a media gesture

# Smoothing factor for volume. Higher value = more smoothing, slower response.
SMOOTHING_ALPHA = 0.2
smoothed_vol_perc = 0 # Stores the smoothed volume percentage (0-100)

# --- Helper Functions ---
def calculate_angle(p1, p2, p3):
    """Returns angle between points (p1-p2-p3) in degrees."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def send_media_key(key):
    """Sends a media key press using pynput."""
    keyboard.press(key)
    keyboard.release(key)

# --- Core Gesture Control Logic (runs in a separate thread) ---
def gesture_control_loop(show_opencv_feed_var):
    global running_gesture_control, cap, system_volume_interface
    global last_play_pause_time, last_next_track_time, last_mute_unmute_time
    global smoothed_vol_perc, volume_freeze_active, volume_freeze_start_time

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        ui_update_queue.put({"type": "error", "message": "Could not open webcam. Please check if it's connected and not in use."})
        running_gesture_control = False
        return

    # --- Camera Property Settings for better tracking ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # 0 for off, 1 for on
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1 for manual, 3 for auto (depends on backend)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6) # Adjust this value (-13 to 0 usually, or absolute values)
        cap.set(cv2.CAP_PROP_GAIN, 0) # Adjust this value (0 to 255 usually)
    except Exception as e:
        print(f"Warning: Could not set camera properties (autofocus/exposure/gain): {e}")

    while running_gesture_control:
        success, img = cap.read()
        if not success:
            ui_update_queue.put({"type": "warning", "message": "Failed to read frame from camera. Stopping control."})
            running_gesture_control = False
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        current_vol_perc_raw = -1
        action_text = ""
        current_time = time.time()

        # Check and clear volume freeze if duration exceeded
        if volume_freeze_active and (current_time - volume_freeze_start_time > VOLUME_FREEZE_DURATION):
            volume_freeze_active = False

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lm = handLms.landmark
                h, w, _ = img.shape

                landmarks = []
                for id, lmk in enumerate(lm):
                    cx, cy = int(lmk.x * w), int(lmk.y * h)
                    landmarks.append((cx, cy))

                # --- Volume Control (Thumb-Index Angle) ---
                if len(landmarks) > 8 and len(landmarks) > 4 and len(landmarks) > 0:
                    p_index = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
                    p_thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP.value]
                    p_wrist = landmarks[mp_hands.HandLandmark.WRIST.value]

                    cv2.circle(img, p_index, 10, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, p_thumb, 10, (255, 0, 255), cv2.FILLED)
                    cv2.line(img, p_index, p_thumb, (255, 0, 255), 3)

                    angle = calculate_angle(p_index, p_thumb, p_wrist)
                    angle = np.clip(angle, 20, 160)

                    current_vol_perc_raw = np.interp(angle, [20, 160], [100, 0])

                    # Only update smoothed_vol_perc and set system volume if not frozen
                    if not volume_freeze_active:
                        smoothed_vol_perc = SMOOTHING_ALPHA * current_vol_perc_raw + (1 - SMOOTHING_ALPHA) * smoothed_vol_perc
                        vol_to_set = int(smoothed_vol_perc)

                        if system_volume_interface:
                            vol = np.interp(vol_to_set, [0, 100], [min_system_vol, max_system_vol])
                            system_volume_interface.SetMasterVolumeLevel(vol, None)
                    else:
                        # If frozen, continue to show the last smoothed value for UI consistency
                        vol_to_set = int(smoothed_vol_perc)


                    volBarHeight = np.interp(vol_to_set, [0, 100], [400, 150])
                    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
                    cv2.rectangle(img, (50, int(volBarHeight)), (85, 400), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, f'{vol_to_set}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # --- Gesture Recognition for Media Control ---

                # Play/Pause (Closed Fist)
                is_fist = False
                if len(landmarks) > 20:
                    thumb_close = np.linalg.norm(np.array(landmarks[mp_hands.HandLandmark.THUMB_TIP.value]) - np.array(landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP.value])) < 50
                    index_curled = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value][1] > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP.value][1]
                    middle_curled = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value][1] > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP.value][1]
                    ring_curled = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP.value][1] > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP.value][1]
                    pinky_curled = landmarks[mp_hands.HandLandmark.PINKY_TIP.value][1] > landmarks[mp_hands.HandLandmark.PINKY_PIP.value][1]

                    if thumb_close and index_curled and middle_curled and ring_curled and pinky_curled:
                        is_fist = True

                if is_fist and (current_time - last_play_pause_time > GESTURE_COOLDOWN):
                    send_media_key(Key.media_play_pause)
                    action_text = "Play/Pause"
                    last_play_pause_time = current_time
                    volume_freeze_active = True # Freeze volume
                    volume_freeze_start_time = current_time

                # Next Track (Index and Pinky Close)
                is_next_gesture = False
                if len(landmarks) > 20:
                    dist_index_pinky = np.linalg.norm(np.array(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]) - np.array(landmarks[mp_hands.HandLandmark.PINKY_TIP.value]))
                    if dist_index_pinky < 35:
                        is_next_gesture = True

                if is_next_gesture and (current_time - last_next_track_time > GESTURE_COOLDOWN):
                    send_media_key(Key.media_next)
                    action_text = "Next Track"
                    last_next_track_time = current_time
                    volume_freeze_active = True # Freeze volume
                    volume_freeze_start_time = current_time

                # Mute/Unmute (Peace Sign / V-sign)
                is_peace_sign = False
                if len(landmarks) > 20:
                    angle_index_straight = calculate_angle(landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP.value],
                                                          landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP.value],
                                                          landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value])

                    angle_middle_straight = calculate_angle(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value],
                                                           landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP.value],
                                                           landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value])

                    angle_ring_bent = calculate_angle(landmarks[mp_hands.HandLandmark.RING_FINGER_MCP.value],
                                                     landmarks[mp_hands.HandLandmark.RING_FINGER_PIP.value],
                                                     landmarks[mp_hands.HandLandmark.RING_FINGER_TIP.value])

                    angle_pinky_bent = calculate_angle(landmarks[mp_hands.HandLandmark.PINKY_MCP.value],
                                                      landmarks[mp_hands.HandLandmark.PINKY_PIP.value],
                                                      landmarks[mp_hands.HandLandmark.PINKY_TIP.value])

                    thumb_distance_to_wrist = np.linalg.norm(np.array(landmarks[mp_hands.HandLandmark.THUMB_TIP.value]) - np.array(landmarks[mp_hands.HandLandmark.WRIST.value]))
                    is_thumb_tucked = thumb_distance_to_wrist < 100

                    if (angle_index_straight > 150 and
                        angle_middle_straight > 150 and
                        angle_ring_bent < 100 and
                        angle_pinky_bent < 100 and
                        is_thumb_tucked):
                        is_peace_sign = True

                if is_peace_sign and (current_time - last_mute_unmute_time > GESTURE_COOLDOWN):
                    send_media_key(Key.media_volume_mute)
                    action_text = "Mute/Unmute"
                    last_mute_unmute_time = current_time
                    volume_freeze_active = True # Freeze volume
                    volume_freeze_start_time = current_time

                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        # Send smoothed_vol_perc (which will be frozen if a media gesture occurred)
        ui_update_queue.put({"type": "update", "volume": int(smoothed_vol_perc), "action": action_text})

        if show_opencv_feed_var.get():
            cv2.imshow("Gesture Control Camera Feed", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running_gesture_control = False
        else:
            if cv2.getWindowProperty("Gesture Control Camera Feed", cv2.WND_PROP_VISIBLE) >= 1:
                 cv2.destroyWindow("Gesture Control Camera Feed")

    if cap:
        cap.release()
    cv2.destroyAllWindows()
    ui_update_queue.put({"type": "stopped"})


# --- GUI Setup ---
class GestureControllerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hand Gesture Media Controller")
        self.geometry("650x550") # Increased height for more modern spacing/elements
        self.resizable(False, False)

        # Set consistent padding
        self.default_padx = 25
        self.default_pady = 15

        # Configure grid for better layout management
        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7), weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        # Main Title/Status
        self.status_label = ctk.CTkLabel(self, text="Initializing...", font=ctk.CTkFont(size=20, weight="bold"))
        self.status_label.grid(row=0, column=0, columnspan=2, pady=(20, 10))

        # Volume Section
        volume_frame = ctk.CTkFrame(self)
        volume_frame.grid(row=1, column=0, columnspan=2, padx=self.default_padx, pady=self.default_pady, sticky="ew")
        volume_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(volume_frame, text="Current Volume", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(5, 0))
        self.volume_label = ctk.CTkLabel(volume_frame, text="--%", font=ctk.CTkFont(size=30, weight="bold"))
        self.volume_label.pack(pady=(0, 5))

        self.volume_slider = ctk.CTkSlider(volume_frame, from_=0, to=100, orientation="horizontal",
                                           command=self.set_volume_from_slider)
        self.volume_slider.set(0)
        self.volume_slider.pack(fill="x", padx=20, pady=(0, 10))


        # Last Action Section
        action_frame = ctk.CTkFrame(self)
        action_frame.grid(row=2, column=0, columnspan=2, padx=self.default_padx, pady=self.default_pady, sticky="ew")
        action_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(action_frame, text="Last Media Action", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(5, 0))
        self.last_action_label = ctk.CTkLabel(action_frame, text="None", font=ctk.CTkFont(size=20))
        self.last_action_label.pack(pady=(0, 5))


        # Controls Section
        controls_frame = ctk.CTkFrame(self)
        controls_frame.grid(row=3, column=0, columnspan=2, padx=self.default_padx, pady=self.default_pady, sticky="ew")
        controls_frame.grid_columnconfigure((0, 1), weight=1) # Two columns for buttons

        self.stop_button = ctk.CTkButton(controls_frame, text="Stop Gesture Control", command=self.stop_control,
                                         font=ctk.CTkFont(size=16, weight="bold"),
                                         fg_color="red", hover_color="darkred")
        self.stop_button.grid(row=0, column=0, columnspan=2, padx=20, pady=10) # Centered in its frame

        self.show_opencv_var = ctk.BooleanVar(value=True)
        self.show_opencv_checkbox = ctk.CTkCheckBox(controls_frame, text="Show Camera Feed",
                                                   variable=self.show_opencv_var,
                                                   font=ctk.CTkFont(size=14))
        self.show_opencv_checkbox.grid(row=1, column=0, columnspan=2, pady=(0, 10))


        # Instructions Section
        self.instructions_label = ctk.CTkLabel(self,
                                                text="Gestures:\n"
                                                     "  - Volume: Twist hand (angle between thumb & index).\n"
                                                     "  - Play/Pause: Make a closed fist.\n"
                                                     "  - Next Track: Bring index & pinky fingers close.\n"
                                                     "  - Mute/Unmute: Make a 'Peace' (V) sign.\n\n"
                                                     "Tip: Ensure good lighting for best tracking.",
                                                font=ctk.CTkFont(size=13),
                                                wraplength=600, justify="left")
        self.instructions_label.grid(row=4, column=0, columnspan=2, pady=(0, self.default_pady), padx=self.default_padx, sticky="w")

        # Dark/Light Mode Toggle
        self.appearance_mode_label = ctk.CTkLabel(self, text="Theme:", font=ctk.CTkFont(size=12))
        self.appearance_mode_label.grid(row=7, column=0, sticky="e", padx=(0, 5))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self, values=["System", "Light", "Dark"],
                                                             command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=7, column=1, sticky="w", padx=(5, self.default_padx))
        self.appearance_mode_optionemenu.set(ctk.get_appearance_mode().capitalize())


        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start gesture control thread immediately
        self.start_control()
        # Start monitoring the queue for updates
        self.after(50, self.update_gui_from_queue)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def start_control(self):
        global running_gesture_control
        if not running_gesture_control:
            running_gesture_control = True
        self.status_label.configure(text="Gesture control active...", text_color="green")
        self.stop_button.configure(state="normal")
        threading.Thread(target=gesture_control_loop, args=(self.show_opencv_var,), daemon=True).start()

    def stop_control(self):
        global running_gesture_control
        if running_gesture_control:
            running_gesture_control = False
            self.status_label.configure(text="Control stopped.", text_color="red")
            self.stop_button.configure(state="disabled")
            self.volume_label.configure(text="--%")
            self.volume_slider.set(0)
            self.last_action_label.configure(text="None", text_color="grey")


    def set_volume_from_slider(self, value):
        if system_volume_interface:
            vol_percentage = float(value)
            vol = np.interp(vol_percentage, [0, 100], [min_system_vol, max_system_vol])
            system_volume_interface.SetMasterVolumeLevel(vol, None)
            self.volume_label.configure(text=f"{int(vol_percentage)}%")
        else:
            self.volume_label.configure(text=f"--% (Audio Unavailable)")


    def update_gui_from_queue(self):
        try:
            while not ui_update_queue.empty():
                message = ui_update_queue.get_nowait()
                if message["type"] == "update":
                    vol_perc = message["volume"]
                    action_text = message["action"]

                    if vol_perc != -1: # Only update if hand was detected in the last frame
                        self.volume_label.configure(text=f"{int(vol_perc)}%")
                        self.volume_slider.set(vol_perc)
                    else:
                        # If no hand detected, and not frozen from last action, display "--%"
                        if not volume_freeze_active: # access global variable in GUI, careful
                           self.volume_label.configure(text="--%")

                    if action_text:
                        self.last_action_label.configure(text=f"{action_text}", text_color="purple")
                        self.after(2000, lambda: self.last_action_label.configure(text="None", text_color="grey"))


                elif message["type"] == "stopped":
                    self.stop_control()
                elif message["type"] == "error":
                    messagebox.showerror("Error", message["message"])
                    self.stop_control()
                elif message["type"] == "warning":
                    messagebox.showwarning("Warning", message["message"])
        except queue.Empty:
            pass

        self.after(50, self.update_gui_from_queue)


    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.stop_control()
            self.destroy()

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = GestureControllerApp()
    app.mainloop()
