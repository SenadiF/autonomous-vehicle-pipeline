import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
#Load pretrained feature extractor
model = VGG16(weights="imagenet", include_top=False)

#load image 
image = cv2.imread("C:/Users/Lenovo/OneDrive/Desktop/autonomous-vehicle-pipeline/perception/image.jpg")
orig = image.copy()
(h, w) = image.shape[:2]

#Regional proposal(Selective Search)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()

rects = ss.process()
#Limit to top 200 proposals for speed
proposals = [] #CNN inputs
boxes = [] #Bounding box coordinates

for (x, y, w_box, h_box) in rects[:200]:
    roi = image[y:y+h_box, x:x+w_box]

    if roi.shape[0] == 0 or roi.shape[1] == 0:
        continue
    #Extract region of interest and preprocess for CNN
    roi = cv2.resize(roi, (224, 224))
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    proposals.append(roi)
    boxes.append((x, y, x+w_box, y+h_box))
proposals = np.array(proposals)

features = model.predict(proposals, verbose=0)

classifier = tf.keras.layers.Dense(10, activation='softmax')
#Predict classes _
preds = classifier(features)

#Filter predictions based on confidence and visualize (Non max suppression - simplified)
for i in range(len(preds)):
    confidence = np.max(preds[i])
    label = np.argmax(preds[i])

    if confidence > 0.9:  # threshold
        (startX, startY, endX, endY) = boxes[i]
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0,255,0), 2)
        cv2.putText(orig, f"{label}:{confidence:.2f}", 
                    (startX, startY-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()