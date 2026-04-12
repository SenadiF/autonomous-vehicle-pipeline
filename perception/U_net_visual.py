import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
#Load and preprocess the image
image = cv2.imread("perception/image.jpg")
image = cv2.resize(image, (224, 224))
image = image / 255.0

input_image = np.expand_dims(image, axis=0)

# Define a simple U-Net architecture

inputs = layers.Input((224,224,3))

# Encoder
c1 = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
p1 = layers.MaxPooling2D((2,2))(c1)

c2 = layers.Conv2D(128, (3,3), padding='same', activation='relu')(p1)
p2 = layers.MaxPooling2D((2,2))(c2)

# Bottleneck
b = layers.Conv2D(256, (3,3), padding='same', activation='relu')(p2)

# Decoder
u1 = layers.Conv2DTranspose(128, (2,2), strides=2, padding='same')(b)
u1 = layers.Concatenate()([u1, c2])

u2 = layers.Conv2DTranspose(64, (2,2), strides=2, padding='same')(u1)
u2 = layers.Concatenate()([u2, c1])

# Output
outputs = layers.Conv2D(3, (1,1), activation='softmax')(u2)

model = Model(inputs, outputs)

#Immediate Model
layer_outputs = [
    c1,   # early features
    p1,   # downsampled
    p2,   # deeper compressed
    b,    # bottleneck
    u1,   # upsampled
    u2,   # near final
    outputs
]

visual_model = Model(inputs=inputs, outputs=layer_outputs)

#Run the image through the model
results = visual_model.predict(input_image)

def show_feature_map(feature_map, title):
    plt.imshow(feature_map[0,:,:,0], cmap='gray')
    plt.title(title)
    plt.show()

titles = [
    "After Conv1",
    "After Pool1",
    "After Pool2",
    "Bottleneck",
    "After Upsample1",
    "After Upsample2",
    "Final Output"
]

for i in range(len(results)):
    show_feature_map(results[i], titles[i])