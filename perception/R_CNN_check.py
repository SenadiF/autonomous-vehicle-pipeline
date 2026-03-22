import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Load model
model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1")

# Read image
image = cv2.imread("C:/Users/Lenovo/OneDrive/Desktop/autonomous-vehicle-pipeline/perception/image.jpg")
image = cv2.resize(image, (640, 640))

# Keep original for visualization
img = image.copy()

# Prepare input
input_image = np.expand_dims(image.astype(np.uint8), axis=0)

# Run detection
outputs = model(input_image)

boxes = outputs['detection_boxes'][0].numpy()
scores = outputs['detection_scores'][0].numpy()
classes = outputs['detection_classes'][0].numpy().astype(int)

h, w, _ = img.shape

for i in range(len(scores)):
    if scores[i] > 0.5:
        ymin, xmin, ymax, xmax = boxes[i]

        x1 = int(xmin * w)
        x2 = int(xmax * w)
        y1 = int(ymin * h)
        y2 = int(ymax * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("R-CNN Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()