import os
from abc import abstractmethod

# from keras.models import load_model
# import keras.backend as K
import chainer.serializers
import glob

import odin
from odin.compute import default_interface as co
from odin.dataset import load_dataset


class LayerWrapper(object):
    """
    Wrapper for layers in a network
    """
    weights = None
    biases = None
    units = None

    """
    The original layer object
    """
    original = None


class ModelWrapper(object):
    model_name = "name_not_specified"
    dataset_name = "dataset_not_specified"
    _saved_model_name = "saved_model.h5"

    def __init__(self, **kwargs):
        self.args = kwargs
        self.prefix = kwargs.get('prefix', None)
        self.dataset = self.load_dataset()
        if len(self.dataset) == 4:
            self.x_train, self.y_train, self.x_test, self.y_test = self.dataset
        self.model = self.load(new_model=kwargs.get("new_model", False))
        self._elements = {}
        self._layers = []

    def get_group(self, group):

        if group not in self._elements.keys():
            datastore = co.load_group(group_name=group, model_wrapper=self)
            self._elements[group] = datastore

        return self._elements[group]

    def get_element(self, group, element_name):
        self.get_group(group)
        return self._elements[group][element_name]

    @abstractmethod
    def load(self, new_model=False):
        raise NotImplemented

    @abstractmethod
    def load_dataset(self):
        raise NotImplemented

    @property
    def model_path(self):
        if self.prefix:
            return os.path.join(odin.model_save_dir, self.model_name, self.prefix)
        else:
            return os.path.join(odin.model_save_dir, self.model_name)

    @property
    def saved_model_path(self):
        return os.path.join(self.model_path, self._saved_model_name)

    @abstractmethod
    def construct(self, **kwargs):
        raise NotImplemented

    @abstractmethod
    def train(self, x_train=None, y_train=None, **options):
        raise NotImplemented

    @abstractmethod
    def save(self):
        raise NotImplemented

    @abstractmethod
    def layers(self):
        raise NotImplemented

    @abstractmethod
    def weights(self):
        raise NotImplemented

    def __str__(self):
        return "%s(%s) {dataset: %s} at '%s'" % (self.model_name, self.__class__.__name__, self.dataset_name, self.model_path)


class KerasModelWrapper(ModelWrapper):

    def weights(self):
        return self.model.get_weights()

    def construct(self, **kwargs):
        pass

    def layers(self):
        pass

    def load(self, new_model=False):
        pass

    model_type = "keras"
    dataset_name = "mnist"

    def save(self):
        if not os.path.isdir(odin.model_save_dir):
            os.makedirs(odin.model_save_dir)
        self.model.save(self.saved_model_path)

        # def get_nth_layer_output(self, n, batch):
        #    if self.model_type == "keras":
        # intermediate_layer_model = Model(inputs=model.input,
        #                                 outputs=layer.output)
        # output = intermediate_layer_model.predict(batch)
        # layer_output = K.function([self.model.layers[0].input],
        #                          [self.model.layers[n].output])
        # output = layer_output([batch])[0]

        # return output

    def train(self, x_train=None, y_train=None, **options):
        x_train = x_train if x_train is not None else self.x_train
        y_train = y_train if y_train is not None else self.y_train

        self.model.fit(x_train, y_train, **options)

    def load_dataset(self):
        return load_dataset(self.dataset_name, options=self.args)


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

    def train(self, x_train=None, y_train=None, **options):
        pass

    def construct(self, **kwargs):
        pass




