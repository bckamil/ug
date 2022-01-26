import tensorflow as tf


class BaseModel(tf.keras.Model):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )
        self.loss_tracker.update_state(loss)
        loss_result = self.loss_tracker.result()
        return {"loss": loss_result}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        loss_result = self.loss_tracker.result()
        return {"loss": loss_result}

    def _compute_loss(self, data):
        raise NotImplementedError("Please Implement this method")

    @property
    def metrics(self):
        return [self.loss_tracker]
