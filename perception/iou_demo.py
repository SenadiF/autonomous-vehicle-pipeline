import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define two boxes
boxA = [50, 50, 200, 200]   # True box
boxB = [100, 100, 250, 250] # Prediction

def compute_iou(boxA, boxB):
    # Intersection coordinates (Overlap starts where both boxes overlap so we take the max but overlap ends at the min )
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Compute areas of boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

iou_value = compute_iou(boxA, boxB)
print("IoU:", iou_value)

# Visualization
image = np.zeros((300, 300, 3), dtype=np.uint8)

# Draw boxes
cv2.rectangle(image, (boxA[0], boxA[1]), (boxA[2], boxA[3]), (0,255,0), 2) # Green
cv2.rectangle(image, (boxB[0], boxB[1]), (boxB[2], boxB[3]), (255,0,0), 2) # Blue

plt.imshow(image)
plt.title(f"IoU = {iou_value:.2f}")
plt.axis('off')
plt.show()