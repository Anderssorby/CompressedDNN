import os
from abc import abstractmethod

# from keras.models import load_model
# import keras.backend as K
import chainer.serializers

import odin
from odin.compute import default_interface as co
from odin.dataset import load_dataset


class ModelWrapper(object):
    model_name = "name_not_specified"

    def __init__(self, **kwargs):
        self.args = kwargs
        self.prefix = kwargs.get('prefix', None)
        self.dataset = self.load_dataset()
        if len(self.dataset) == 4:
            self.x_train, self.y_train, self.x_test, self.y_test = self.dataset
        if os.path.isfile(self.model_path):
            self.load()
        else:
            self.model = self.construct()
        # self.model.model_name = self.model_name
        self._elements = {}

    def get_group(self, group):

        if group not in self._elements.keys():
            datastore = co.load_group(group_name=group, model_wrapper=self)
            self._elements[group] = datastore

        return self._elements[group]

    def get_element(self, group, element_name):
        self.get_group(group)
        return self._elements[group][element_name]

    @abstractmethod
    def load(self):
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


class KerasModelWrapper(ModelWrapper):

    def weights(self):
        return self.model.get_weights()

    def construct(self, **kwargs):
        pass

    def layers(self):
        pass

    def load(self):
        pass

    model_type = "keras"
    dataset_name = "mnist"

    def save(self):
        if not os.path.isdir(odin.model_save_dir):
            os.makedirs(odin.model_save_dir)
        self.model.save(self.model_path)

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


class ChainerModelWrapper(ModelWrapper):
    model_type = "chainer"
    dataset_name = "mnist"

    def load(self):

        if os.path.isdir(self.model_path):
            snapshot = os.path.join(self.model_path, "snapshot_iter_1")
        else:
            snapshot = self.model_path
        chainer.serializers.load_hdf5(snapshot,
                                      self.construct())

    def load_dataset(self):
        return load_dataset(self.dataset_name, options=self.args)

    def layers(self):
        # TODO debug!!!!
        _layers = []
        for c in self.model.predictor.children():
            _layers.append(c)

        return _layers

    def save(self):
        pass

    def weights(self):
        pass

    def transfer_to_architecture(self, layer_widths):

        layers = self.layers()
        weights = []
        biases = []
        for layer in layers:
            if type(layer) == "chainer.links.Linear":
                weights.append(layer.W)
                biases.append(layer.b)

        self.construct(layer_widths=layer_widths, weights=weights, biases=biases)

    def train(self, x_train=None, y_train=None, **options):
        pass

    def construct(self, **kwargs):
        pass




