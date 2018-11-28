import os

import odin
from odin.dataset import load_dataset
from odin.models.base import ModelWrapper, LayerWrapper
import keras.models
from keras import backend as K


class KerasLayer(LayerWrapper):
    layer_types = {
        "keras.layers.Dense": "fully_connected",
        "keras.layers.Conv2D": "convolution"
    }

    def __init__(self, layer):
        self.type = self.layer_types.get(type(layer).__name__, "unknown")
        self.weights = layer.W
        self.biases = layer.b
        self.units = layer.out_size
        self.original = layer


class KerasModelWrapper(ModelWrapper):
    model_type = "keras"

    def weights(self):
        return self.model.get_weights()

    def layers(self):
        pass

    def load(self, new_model=False):
        if new_model:
            return self.construct()
        else:
            return keras.models.load_model(self.saved_model_path)

    def save(self):
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.model.save(self.saved_model_path)

    def get_nth_layer_output(self, n, x):
        out_function = K.function([self.model.layers[0].input],
                                          [self.model.layers[n].output])

        layer_output = out_function([x])[0]
        return layer_output

    def get_layer_outputs(self, x):
        num_layers = len(self.model.layers)
        layer_outputs = []
        for l in range(num_layers):
            layer_outputs.append(self.get_nth_layer_output(l, x))

        return layer_outputs

    def train(self, x_train=None, y_train=None, **options):
        x_train = x_train if x_train is not None else self.x_train
        y_train = y_train if y_train is not None else self.y_train

        self.model.fit(x_train, y_train, **options)

    def test(self):
        (_, _), (x_test, y_test) = self.load_dataset()

        scores = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def load_dataset(self):
        return load_dataset(self.dataset_name, options=self.args)