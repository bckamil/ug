import tensorflow as tf

from ug.models.base_model import BaseModel


class TripletModel(BaseModel):
    def __init__(self, network, alpha=0.5):
        super(TripletModel, self).__init__()
        self.network = network
        self.alpha = alpha

    def _compute_loss(self, data):
        ap_distance, an_distance = self.network(data)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.alpha, 0.0)
        return loss
