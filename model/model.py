import tensorflow as tf
from keras.layers import Concatenate
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from keras import regularizers


def define_model_architecture(input_shape=(None, None, 1, 5)):
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # encoder.. first half of model, will be starting with convolutional layers
    x = tf.keras.layers.Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                               kernel_regularizer=regularizers.l2(0.01))(input_layer)  # filters more dep on input shape
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                               kernel_regularizer=regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # bottleneck
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same',
                               kernel_regularizer=regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # decoder
    x = tf.keras.layers.Conv3DTranspose(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                        kernel_regularizer=regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Conv3D(filters=2, kernel_size=(1, 1, 1), activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer='adam', loss=BinaryCrossentropy())
    print(model.summary())
    return model
