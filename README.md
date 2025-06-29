🖐️ Hand Gesture Media Controller

Control your computer's media playback and volume with intuitive hand gestures! This Python-based application leverages real-time hand tracking through your webcam to provide a touch-free and modern way to interact with your music, videos, and system volume.

✨ Features

    Effortless Volume Control: Adjust your system volume by simply twisting your hand (changing the angle between your thumb and index finger).

    Touch-Free Media Playback:

        Play/Pause: Make a closed fist to toggle playback.

        Next Track: Bring your index and pinky fingers close together to skip to the next song.

        Mute/Unmute: Perform a 'Peace' (V) sign to instantly mute or unmute your audio.

    Smooth & Stable Tracking: Enhanced MediaPipe configurations and a smart volume "freeze" mechanism prevent unintended volume jumps when performing other gestures.

    Modern User Interface: A clean, intuitive GUI built with CustomTkinter, featuring real-time volume display, last action feedback, and even a dark mode toggle.

    Real-time Camera Feed: Option to display a live feed of your hand tracking for visual feedback and debugging.

    Automatic Startup: The application starts gesture control immediately upon launch.

🛠️ Installation

Before you begin, ensure you have Python 3.8 or higher installed on your system.

    Clone the Repository:
    Bash

git clone https://github.com/YourGitHubUsername/Hand-Gesture-Media-Controller.git
cd Hand-Gesture-Media-Controller

(Replace YourGitHubUsername/Hand-Gesture-Media-Controller with your actual repository path)

Create a Virtual Environment (Recommended):
Bash

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install Dependencies:
Bash

    pip install opencv-python mediapipe numpy pynput pycaw customtkinter

        opencv-python: For webcam access and image processing.

        mediapipe: Google's powerful framework for hand landmark detection.

        numpy: For numerical operations, especially with angles and interpolations.

        pynput: To simulate keyboard media key presses (Play/Pause, Next, Mute).

        pycaw: To control the system volume on Windows.

        customtkinter: For the modern GUI.

🚀 Usage

    Run the Application:
    Once all dependencies are installed, simply run the main Python script:
    Bash

    python your_main_script_name.py

    (If your main script is named something other than main.py or app.py, adjust the command accordingly)

    Grant Camera Access:
    Your operating system might prompt you to grant camera access to the application. Please allow it.

    Start Gesturing!
    The application will launch, open your webcam, and begin listening for gestures.

        Volume: Position your hand clearly in front of the camera. Twist your hand (changing the angle between your thumb and index finger) to adjust the volume.

        Play/Pause: Form a tight closed fist.

        Next Track: Bring your index finger tip and pinky finger tip very close together.

        Mute/Unmute: Form a clear 'Peace' (V) sign with your index and middle fingers extended, and other fingers curled.

    Stop Control:
    Click the "Stop Gesture Control" button in the GUI, or simply close the camera feed window (if displayed) and press 'Q' in that window.

⚙️ Configuration & Customization

The core logic is within gesture_control_loop. You can modify several parameters to fine-tune performance and gesture detection:

    GESTURE_COOLDOWN (Line ~50): Adjust the time (in seconds) between detecting the same media gesture to prevent rapid, accidental re-triggers.

    SMOOTHING_ALPHA (Line ~56): Controls the responsiveness vs. smoothness of volume control.

        Higher value (closer to 1.0): More responsive, less smooth.

        Lower value (closer to 0.0): Smoother, less responsive.

    VOLUME_FREEZE_DURATION (Line ~54): The time (in seconds) the volume remains frozen after a media control gesture. Increase if you notice volume jumps, decrease if it feels too unresponsive.

    min_detection_confidence & min_tracking_confidence (Line ~41): MediaPipe's confidence thresholds.

        Increase for stricter detection (fewer false positives).

        Decrease for more lenient detection (may pick up more hands, but also more noise).

    Camera Properties (cv2.CAP_PROP_AUTOFOCUS, _EXPOSURE, _GAIN):
    These lines (around Line ~120 in gesture_control_loop) attempt to set manual camera properties for more stable tracking. These values (-6 for exposure, 0 for gain) are highly dependent on your specific webcam and lighting conditions. If tracking is inconsistent, experiment with commenting these out or adjusting their values. Not all webcams support these manual controls.

    Gesture Thresholds: The numerical thresholds used in the is_fist, is_next_gesture, and is_peace_sign logic (e.g., dist < 50, angle > 150) are crucial. You will likely need to adjust these values by observing the live camera feed and understanding your hand's typical shape for each gesture. Add print() statements in the gesture_control_loop to see the actual distances/angles your hand produces.

💡 Troubleshooting

    "Could not open webcam" error:

        Ensure your webcam is connected and not being used by another application.

        Check your operating system's privacy settings to ensure applications are allowed to access the camera.

        Try restarting your computer.

    Gestures not recognized or too sensitive:

        Lighting is key! Ensure even, bright lighting on your hand, without strong backlighting or shadows.

        Adjust the min_detection_confidence, min_tracking_confidence, and especially the gesture-specific numerical thresholds as described in the "Configuration & Customization" section.

        Ensure your hand is clearly visible and within the camera's frame.

    Volume jumps when performing other gestures:

        Increase VOLUME_FREEZE_DURATION.

        Adjust SMOOTHING_ALPHA to a lower value for more smoothing.

    PyCaw error on startup:

        pycaw is Windows-specific. If you're on macOS or Linux, the volume control part won't work, but media controls (Play/Pause, Next, Mute) via pynput should still function. The error indicates pycaw couldn't access your audio device, which might happen if no audio output device is active.

🙏 Acknowledgments

    MediaPipe Hands for robust hand tracking.

    CustomTkinter for the modern GUI elements.

    PyCaw for Windows audio control.

    Pynput for keyboard simulation.
