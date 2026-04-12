import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

#load image
img_path='C:/Users/Lenovo/OneDrive/Desktop/autonomous-vehicle-pipeline/perception/image.jpg'
image=cv2.imread(img_path)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_resized=cv2.resize(image,(224,224))

#Preprocess for VGG16  (13 convolutional layers and 3 fully connected layers)
image_input = np.expand_dims(image_resized, axis=0)#aaded another dimension to make it compatible with the model input
image_input = tf.keras.applications.vgg16.preprocess_input(image_input)

#Load pre-trained VGG16 model
base_model=tf.keras.applications.VGG16(weights='imagenet',include_top=False)

#Choose a layer to visualize
layer_name='block1_conv1'
layer_output=base_model.get_layer(layer_name).output

#Create a model that outputs the intermediate activations
model=tf.keras.Model(inputs=base_model.input,outputs=layer_output)

#Get feature maps
feature_maps=model.predict(image_input)

#plot feature maps
num_filters=feature_maps.shape[-1]
plt.figure(figsize=(12,12))

for i in range(min(16, num_filters)):
    plt.subplot(4, 4, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap='gray')
    plt.axis('off')

plt.suptitle("Feature Maps from VGG16 (block1_conv1)")
plt.show()