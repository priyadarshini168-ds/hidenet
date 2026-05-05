import tensorflow as tf
from tensorflow.keras import layers, Model

def build_detector(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape, name='detector_input')

    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64)(x)
    x = layers.Activation('relu')(x)

    output = layers.Dense(1, activation='sigmoid', name='detector_output')(x)

    model = Model(inputs=inputs, outputs=output, name='SteganalysisDetector')
    return model

if __name__ == "__main__":
    model = build_detector()
    model.summary()
    print("Steganalysis Detector built successfully!")