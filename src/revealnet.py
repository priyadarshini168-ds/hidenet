import tensorflow as tf
from tensorflow.keras import layers, Model

def build_revealnet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape, name='stego_input')

    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    output = layers.Conv2D(3, 1, activation='sigmoid', name='revealed_output')(x)

    model = Model(inputs=inputs, outputs=output, name='RevealNet')
    return model

if __name__ == "__main__":
    model = build_revealnet()
    model.summary()
    print("RevealNet built successfully!")