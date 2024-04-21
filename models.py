import tensorflow as tf
from tensorflow import keras

def generateModel1(class_size : int):
    model1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(224, 224, 3), kernel_size=(3,3), filters=32, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', strides=1, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', strides=1, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=class_size, activation='softmax')
    ])
    return model1

def genereateModel2(class_size : int):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(224, 224, 3), kernel_size=(3,3), filters=32, strides=1, padding='same', activation='relu'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32, padding='same', strides=1, activation='relu'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(rate=0.25),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', strides=1, activation='relu'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', strides=1, activation='relu'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(rate=0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=class_size, activation='softmax')
])