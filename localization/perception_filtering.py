import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize Kalman Filter 
# 4 state varibales: [x, y, vx, vy]
# 2 measurement variables: [x, y]
kf = cv2.KalmanFilter(4, 2)
#defines motion 
#x = x + vx  ( keeps moving with the same velocity)
#y = y + vy
kf.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)
# we only measure position, not velocity
kf.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)
#Process noise - uncertainty in the motion
#Measurement noise - uncertainty in the detection
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

# Open video (0 = webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Default measurement (if no detection)
    measurement = None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            #Extract the boxes and convert to centers 
            # This become our measurements
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])

            # Draw detection box (blue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            break  # track only one object - verison one 

    # ---- KALMAN FILTER ----

    # Predict step
    prediction = kf.predict()
    pred_x, pred_y = int(prediction[0]), int(prediction[1])

    # Update step (only if detection exists)
    if measurement is not None:
        kf.correct(measurement)

    # Draw prediction (green)
    cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)

    cv2.imshow("YOLO + Kalman Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()