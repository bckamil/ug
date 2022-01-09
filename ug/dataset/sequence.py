import math
import numpy as np

from tensorflow.keras.utils import Sequence


class DatasetSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, n, generator):
        self.x_set, self.y_set = x_set, y_set
        self.batch_size = batch_size
        self.n = n
        self.generator = generator

        self.x, self.y = self.generator(self.x_set, self.y_set)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        self.epoch += 1

        if self.epoch % self.n == 0:
            self.x, self.y = self.generator(self.x_set, self.y_set)
