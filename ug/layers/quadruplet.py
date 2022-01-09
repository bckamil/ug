import tensorflow as tf
from tensorflow.keras import layers


class QuadrupletDistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, the anchor embedding and the
    first negative embedding and the first negative embedding and the
    second negative embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative, negative_2):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        nn_distance = tf.reduce_sum(tf.square(negative, negative_2), -1)
        return ap_distance, an_distance, nn_distance

