import os

from odin.misc.dataset import load_dataset
from odin.models.base import ModelWrapper, LayerWrapper
import keras.models
from keras import backend
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot


class KerasLayer(LayerWrapper):
    layer_types = {
        "Dense": "fully_connected",
        "Conv2D": "convolution"
    }

    def __init__(self, layer):
        self.type = self.layer_types.get(type(layer).__name__, "unknown")

        if self.type is not "unknown":
            self.weights = layer.get_weights()[0]
            self.biases = layer.bias
            self.output_shape = layer.output_shape
            self.units = layer.output_shape[-1]
        self.original = layer


class KerasModelWrapper(ModelWrapper):
    model_type = "keras"
    history = None

    def weights(self):
        return self.model.get_weights()

    def layers(self, force_update=False):
        if force_update or not self._layers:
            self._layers = []
            for c in self.model.layers:
                layer = KerasLayer(c)
                if layer.type is not "unknown":
                    self._layers.append(layer)

        return self._layers

    def load(self):
        return keras.models.load_model(self.saved_model_path)

    def save(self):
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.model.save(self.saved_model_path)

    def get_nth_layer_output(self, n, x):
        layers = self.layers()
        out_function = backend.function([layers[0].original.input],
                                        [layers[n].original.output])

        layer_output = out_function([x])[0]
        return layer_output

    def get_layer_outputs(self, x):
        layers = self.layers()
        num_layers = len(layers)
        layer_outputs = []
        for l in range(num_layers):
            layer_outputs.append(self.get_nth_layer_output(l, x))

        return layer_outputs

    def train(self, **options):
        (x_train, y_train), (_, _) = self.load_dataset()
        epochs = options.get("epochs", 200)

        self.history = self.model.fit(x_train, y_train, epochs=epochs, verbose=1)

    def test(self):
        (_, _), (x_test, y_test) = self.load_dataset()

        scores = self.model.test_on_batch(x_test, y_test)
        return scores

    def load_dataset(self):
        return load_dataset(self.dataset_name, options=self.args)

    def show(self, format="svg"):
        if format == "png":
            return plot_model(self.model, show_shapes=True, to_file=self.model_name + ".png")
        elif format == "svg":
            dot = model_to_dot(self.model, show_shapes=True).create(prog='dot', format='svg')
            return dot

    def summary(self):
        return self.model.summary()
