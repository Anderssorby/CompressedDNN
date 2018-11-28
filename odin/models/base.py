
import os
from abc import abstractmethod

# from keras.models import load_model
# import keras.backend as K

import odin
from odin.compute import default_interface as co


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
    """
    Base class for all models. Supposed to be implementation and framework independent.
    """

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
            data_store = co.load_group(group_name=group, model_wrapper=self)
            self._elements[group] = data_store

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
    def layers(self) -> [LayerWrapper]:
        raise NotImplemented

    @abstractmethod
    def get_layer_outputs(self, x):
        raise NotImplemented

    @abstractmethod
    def weights(self):
        raise NotImplemented

    def __str__(self):
        return "%s(%s) {dataset: %s} at '%s'" % (self.model_name, self.__class__.__name__, self.dataset_name, self.model_path)




