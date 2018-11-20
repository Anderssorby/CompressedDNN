import glob
import os

import chainer.serializers

from odin.dataset import load_dataset
from odin.models.base import LayerWrapper, ModelWrapper


class ChainerLayer(LayerWrapper):
    layer_types = {
        "chainer.links.connection.linear.Linear": "fully_connected"
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

    def load(self, new_model=False):
        model = self.construct(**self.args)
        snapshot = None

        if os.path.isdir(self.model_path):
            file_filter = os.path.join(self.model_path, "snapshot_iter_*")
            snapshots = glob.glob(file_filter)
            if snapshots:
                snapshot = snapshots[-1]
            else:
                new_model = True
        else:
            new_model = True

        if not new_model:
            path = os.path.join(self.model_path, self._saved_model_name)
            model = chainer.serializers.load_hdf5(path, model)

        return model

    def load_dataset(self):
        return load_dataset(self.dataset_name, options=self.args)

    def layers(self):
        if not self._layers:
            for c in self.model.predictor.children():
                layer = ChainerLayer(c)
                self._layers.append(layer)

        return self._layers

    def save(self):
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
        chainer.serializers.save_hdf5(self.saved_model_path, self.model)

    def weights(self):
        pass

    def get_layer_outputs(self, x):
        return self.model.predictor(x, multi_layer=True)

