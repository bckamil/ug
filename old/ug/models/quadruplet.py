import tensorflow as tf

from ug.models.base_model import BaseModel


class QuardupletModel(BaseModel):
    def __init__(self, network, alpha=0.5, beta=0.5):
        super(QuardupletModel, self).__init__()
        self.network = network
        self.alpha = alpha
        self.beta = beta

    def _compute_loss(self, data):
        ap_distance, an_distance, nn_distance = self.network(data)

        ap_an = tf.maximum(ap_distance - an_distance + self.alpha, 0)
        ap_nn = tf.maximum(ap_distance - nn_distance + self.beta, 0)
        return ap_an + ap_nn
