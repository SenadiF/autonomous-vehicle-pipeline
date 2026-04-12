from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load pretrained YOLO model 
model = YOLO("yolov8n.pt")

# Load image
image_path = "C:/Users/Lenovo/OneDrive/Desktop/autonomous-vehicle-pipeline/perception/image.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run inference
results = model(image_rgb)

# Plot results
annotated_frame = results[0].plot()

# Show image
plt.imshow(annotated_frame)
plt.axis('off')
plt.show()
print(results[0].boxes)