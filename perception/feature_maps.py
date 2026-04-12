import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load image 
image=cv2.imread('C:/Users/Lenovo/OneDrive/Desktop/autonomous-vehicle-pipeline/perception/image.jpg')
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#Convert to grayscale
gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

#Define filters 

#Edge detection kernal(horizontal)
kernel_horizontal=np.array([[-1,-1,-1],
                             [0,0,0],
                             [1,1,1]])

#Edge detection kernal(vertical)
kernel_vertical=np.array([[-1,0,1],
                           [-1,0,1],
                           [-1,0,1]])

#Apply filters using convolution 
feature_map_horizontal=cv2.filter2D(gray,-1,kernel_horizontal)
feature_map_vertical=cv2.filter2D(gray,-1,kernel_vertical)

#Visualization
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Horizontal Edges")
plt.imshow(feature_map_horizontal,cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Vertical Edges")
plt.imshow(feature_map_vertical,cmap='gray')
plt.axis('off')

plt.show()


