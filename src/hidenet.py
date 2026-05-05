import tensorflow as tf
from tensorflow.keras import layers, Model

def attention_gate(x, g, filters):
    theta_x = layers.Conv2D(filters, 1, strides=1, padding='same')(x)
    phi_g = layers.Conv2D(filters, 1, strides=1, padding='same')(g)
    add = layers.Add()([theta_x, phi_g])
    relu = layers.Activation('relu')(add)
    psi = layers.Conv2D(1, 1, strides=1, padding='same')(relu)
    sigmoid = layers.Activation('sigmoid')(psi)
    return layers.Multiply()([x, sigmoid])

def encoder_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def decoder_block(x, skip, filters):
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(filters, 2, padding='same')(x)
    skip = attention_gate(skip, x, filters // 2)
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_hidenet(input_shape=(256, 256, 3)):
    cover_input = layers.Input(shape=input_shape, name='cover_input')
    secret_input = layers.Input(shape=input_shape, name='secret_input')

    combined = layers.Concatenate()([cover_input, secret_input])

    e1 = encoder_block(combined, 64)
    p1 = layers.MaxPooling2D((2, 2))(e1)

    e2 = encoder_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(e2)

    e3 = encoder_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(e3)

    e4 = encoder_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(e4)

    bottleneck = encoder_block(p4, 1024)

    d1 = decoder_block(bottleneck, e4, 512)
    d2 = decoder_block(d1, e3, 256)
    d3 = decoder_block(d2, e2, 128)
    d4 = decoder_block(d3, e1, 64)

    output = layers.Conv2D(3, 1, activation='sigmoid', name='stego_output')(d4)

    model = Model(inputs=[cover_input, secret_input], outputs=output, name='HideNet')
    return model

if __name__ == "__main__":
    model = build_hidenet()
    model.summary()
    print("HideNet built successfully!")