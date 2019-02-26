import glob
import os

import chainer.serializers

from odin.dataset import load_dataset
from odin.models.base import LayerWrapper, ModelWrapper


class ChainerLayer(LayerWrapper):
    layer_types = {
        "Linear": "fully_connected",
        "Conv2D": "convolution_2d"
    }

    def __init__(self, layer):
        self.type = self.layer_types.get(type(layer).__name__, "unknown")
        self.weights = layer.W
        self.biases = layer.b
        self.units = layer.out_size
        self.original = layer


class ChainerModelWrapper(ModelWrapper):
    model_type = "chainer"
    dataset_name = "mnist"

    def load(self):
        model = self.construct(**self.args)

        path = os.path.join(self.model_path, self._saved_model_name)
        chainer.serializers.load_hdf5(path, model)

        return model

    def load_dataset(self):
        return load_dataset(self.dataset_name, options=self.args)

    def layers(self, force_update=False) -> [ChainerLayer]:
        if force_update or not self._layers:
            self._layers = []
            for c in self.model.predictor.children():
                layer = ChainerLayer(c)
                self._layers.append(layer)

        return self._layers

    def save(self):
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
        chainer.serializers.save_hdf5(self.saved_model_path, self.model)

    def weights(self):
        return [l.weights for l in self.layers()]

    def get_layer_outputs(self, x):
        return self.model.predictor(x, multi_layer=True)

    def summary(self):
        return self.model.summary()

