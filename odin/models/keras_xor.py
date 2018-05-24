import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from .base import KerasModelWrapper


class KerasXOR(KerasModelWrapper):
    model_name = 'keras_xor'

    # the four different states of the XOR gate
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

    # the four expected results in the same order
    y_train = np.array([[0], [1], [1], [0]], "float32")

    def construct(self):
        model = Sequential()
        model.add(Dense(16, input_dim=2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='mean_squared_error',
                           optimizer='adam',
                           metrics=['binary_accuracy'])

        self.model = model

    def train(self, x_train=None, y_train=None, **options):
        super(KerasXOR, self).train(nb_epoch=500, **options)


