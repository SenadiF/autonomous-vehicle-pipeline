import tensorflow as tf
from tensorflow.keras import layers, Model

#encode block
# Conv- Extracr features 
# Maxpooling - Reduces resolution 
#Model learns - shapes , objects , patterns
def encoder_block(x, filters):
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    p = layers.MaxPooling2D((2,2))(x) 
    return x, p

#decode block
#unsampling - Increases resolution
#Combine with earlier features from encoder 
def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2,2), strides=2, padding='same')(x)  # ↑ increases size
    x = layers.Concatenate()([x, skip])  
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    return x

def build_unet(input_shape=(224,224,3), num_classes=3):
    inputs = layers.Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)   # 224 → 112
    s2, p2 = encoder_block(p1, 128)      # 112 → 56
    s3, p3 = encoder_block(p2, 256)      # 56 → 28

    # Bottleneck -Means the smallest , mosty compressed respresentation of the image
    #Forces the model to learn the most important features and discard noise
    b1 = layers.Conv2D(512, (3,3), padding='same', activation='relu')(p3)

    # Decoder
    d1 = decoder_block(b1, s3, 256)      # 28 → 56
    d2 = decoder_block(d1, s2, 128)      # 56 → 112
    d3 = decoder_block(d2, s1, 64)       # 112 → 224

    # Output layer
    outputs = layers.Conv2D(num_classes, (1,1), activation='softmax')(d3)

    model = Model(inputs, outputs)
    return model
model = build_unet()
model.summary()