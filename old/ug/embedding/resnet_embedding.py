import tensorflow as tf


def get_rn50_embedding(input_shape: int):
    """
    Return embedding model.
    :param input_shape: Input shape
    """
    rn50 = tf.keras.applications.resnet.ResNet50(
        weights="imagenet", input_shape=input_shape + (3,), include_top=False
    )

    flatten = tf.keras.layers.Flatten()(rn50.output)
    dense1 = tf.keras.layers.Dense(512, activation="relu")(flatten)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = tf.keras.layers.Dense(384, activation="relu")(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense3 = tf.keras.layers.Dense(256, activation="relu")(dense2)
    dense3 = tf.keras.layers.BatchNormalization()(dense3)
    output = tf.keras.layers.Dense(128)(dense3)

    embedding = tf.keras.Model(rn50.input, output)

    trainable = False
    for layer in rn50.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable
    return embedding
