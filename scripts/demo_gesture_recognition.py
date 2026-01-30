"""
Gesture Recognition Demo Script for OM1.
Works with MediaPipe 0.10.32+
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

# Download gesture recognizer model if not exists
MODEL_PATH = "gesture_recognizer.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading gesture recognition model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")

def main():
    print("\n" + "=" * 60)
    print("  OM1 Gesture Recognition Demo")
    print("=" * 60)
    print("\nPress 'q' to quit\n")

    download_model()

    # Setup gesture recognizer
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {width}x{height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = recognizer.recognize(mp_image)

        cv2.rectangle(frame, (0, 0), (width, 60), (50, 50, 50), -1)
        cv2.putText(frame, "OM1 GestureRecognitionInput Demo", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if result.gestures:
            for idx, (gesture, handedness) in enumerate(zip(result.gestures, result.handedness)):
                gesture_name = gesture[0].category_name
                confidence = gesture[0].score
                hand = handedness[0].category_name

                box_y = 80 + idx * 70
                cv2.rectangle(frame, (10, box_y), (450, box_y + 60), (0, 100, 0), -1)
                cv2.putText(frame, f"{hand} Hand: {gesture_name.upper()}",
                            (20, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {confidence:.0%}",
                            (20, box_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                print(f"\nINPUT: Gesture Detector")
                print(f"// START")
                print(f'You see a person making a "{gesture_name}" gesture with their {hand.lower()} hand.')
                print(f"// END")
        else:
            cv2.putText(frame, "Show a hand gesture!", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

        cv2.imshow("OM1 Gesture Recognition Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo ended!")

if __name__ == "__main__":
    main()
