import sys
import threading
import time
import os
import sounddevice as sd
import numpy as np
import whisper
from scipy.io.wavfile import write
from scipy.signal import resample
sys.path.append('pose_detection/')
from pose_detection.gesture_system import GestureProcessor, GestureSystem
from pose_detection.gesture_extractor import PointingGestureDetector
from speech_recog.speech_recog import transcribe_live


def receive_gesture():
    """
    Runs gesture detection while a human is present.
    Stops when the human leaves.
    """
    print("🔍 Checking for human presence...")
    detector = PointingGestureDetector()

    gesture_processor = GestureProcessor(
        detector=detector,
        image_path="./spot_util/hand_color_image.png",
        depth_path="./spot_util/hand_depth_image.png",
        transform_path="./spot_util/hand_intrinsics_and_transform.json"
    )

    GestureSystem(gesture_processor).run()

def transcribe_speech():
    """
    Starts live speech transcription.
    """
    try:
        transcribe_live()
    except Exception as e:
        print(f"❌ Speech transcription error: {e}")


# Remove old gesture files
for path in ["./.tmp/gesture_color.png", "./.tmp/gesture_depth.png",
             "./.tmp/gesture_transformation.json", "./.tmp/gesture_vector.json"]:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

# Create and start threads
gesture_thread = threading.Thread(target=receive_gesture, daemon=True)
audio_thread = threading.Thread(target=transcribe_speech, daemon=True)

gesture_thread.start()
audio_thread.start()

# Keep main thread alive
gesture_thread.join()
audio_thread.join()

print("RECEIVED AUDIO/GESTURE SIGNAL")