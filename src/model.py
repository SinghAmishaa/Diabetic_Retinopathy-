import tensorflow as tf
from tensorflow.keras import layers, models

from config import IMAGE_SIZE, NUM_CLASSES


def build_cnn():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(224,224,3)),

        tf.keras.layers.Conv2D(32,3,activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64,3,activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128,3,activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256,activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(NUM_CLASSES,activation="softmax")

    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(1e-4),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model