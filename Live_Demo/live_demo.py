# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:05:37 2026

@author: asimm
"""

import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# Model Paths
# =========================
cnn_path = "models/baseline_cnn_driver_distraction.keras"
resnet_frozen_path = resnet_frozen_path = "models/resnet50_driver_distraction.keras"
resnet_finetuned_path = resnet_finetuned_path = "models/resnet50_finetuned_driver_distraction.keras"

# =========================
# Load Models
# =========================
cnn_model = load_model(cnn_path, compile=False)
resnet_model = load_model(resnet_frozen_path, compile=False)
resnet_ft_model = load_model(resnet_finetuned_path, compile=False)

# =========================
# Class Names
# =========================
class_names = [
    "drinking",
    "hair and makeup",
    "operating the radio",
    "reaching behind",
    "safe driving",
    "talking on the phone - left",
    "talking on the phone - right",
    "talking to passenger",
    "texting - left",
    "texting - right"
]

# Classes considered distracting
distracted_classes = {
    "drinking",
    "hair and makeup",
    "operating the radio",
    "reaching behind",
    "talking on the phone - left",
    "talking on the phone - right",
    "talking to passenger",
    "texting - left",
    "texting - right"
}

# =========================
# Choose Model
# =========================
print("Choose Model:")
print("1 - CNN")
print("2 - ResNet50 (Frozen)")
print("3 - ResNet50 (Fine-Tuned)")

choice = input("Enter choice (1/2/3): ").strip()

if choice == "1":
    model = cnn_model
    model_name = "CNN"
elif choice == "2":
    model = resnet_model
    model_name = "ResNet50 (Frozen)"
elif choice == "3":
    model = resnet_ft_model
    model_name = "ResNet50 (Fine-Tuned)"
else:
    raise SystemExit("Invalid choice.")

print(f"Using model: {model_name}")

# =========================
# Demo Settings
# =========================
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.50
SMOOTHING_WINDOW = 5
ALERT_FRAME_THRESHOLD = 8   # consecutive confident distracted frames
FRAME_SKIP = 2              # predict every 2 frames for smoother speed

pred_buffer = deque(maxlen=SMOOTHING_WINDOW)
distracted_counter = 0
frame_count = 0
last_avg_pred = None

# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Error: Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror view for a natural webcam display
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Center crop box
    box_size = int(min(h, w) * 0.7)
    start_x = (w - box_size) // 2
    start_y = (h - box_size) // 2
    end_x = start_x + box_size
    end_y = start_y + box_size

    roi = frame[start_y:end_y, start_x:end_x]

    # Predict every FRAME_SKIP frames
    if frame_count % FRAME_SKIP == 0:
        img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)[0]
        pred_buffer.append(pred)

        if len(pred_buffer) > 0:
            last_avg_pred = np.mean(pred_buffer, axis=0)

    frame_count += 1

    if last_avg_pred is not None:
        pred_index = np.argmax(last_avg_pred)
        pred_class = class_names[pred_index]
        confidence = float(last_avg_pred[pred_index])

        if confidence < CONFIDENCE_THRESHOLD:
            status_text = "."  # uncertin
            status_color = (0, 255, 255)
            distracted_counter = max(0, distracted_counter - 1)
        else:
            if pred_class in distracted_classes:
                distracted_counter += 1
                status_text = f"Distracted: {pred_class}"
                status_color = (0, 0, 255)
            else:
                distracted_counter = 0
                status_text = "Safe Driving"
                status_color = (0, 255, 0)

        # Alert logic
        if distracted_counter >= ALERT_FRAME_THRESHOLD:
            alert_text = "ALERT: DISTRACTED DRIVER"
            cv2.putText(
                frame, alert_text, (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3
            )
            # Optional console beep
            print("\a", end="")

        # Main prediction text
        cv2.putText(
            frame, f"Model: {model_name}", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
        )

        cv2.putText(
            frame, status_text, (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2
        )

        cv2.putText(
            frame, f"Confidence: {confidence:.2f}", (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
        )

    # Draw crop box
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 0), 2)
    cv2.putText(
        frame, "Position yourself inside the box", (start_x, start_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
    )

    cv2.imshow("Driver Distraction Detection Live Demo", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()