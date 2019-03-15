import os
from abc import abstractmethod

# from keras.models import load_model
# import keras.backend as K

import odin
from odin.compute import default_interface as co
from odin.utils import dynamic_class_import
import abc


class LayerWrapper(object):
    """
    Wrapper for layers in a network
    """
    type = None
    weights = None
    biases = None
    units = None

    """
    The original layer object
    """
    original = None

    def __str__(self):
        return self.__class__.__name__ + " for type=%s, (%s)" % (self.type, self.original)


class CallbackManager(abc.ABC):
    callbacks: list
    model = None
    params: dict = {}

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def set_model(self, model):
        self.model = model

    def set_params(self, params: dict):
        self.params = params

    def add_callbacks(self, callbacks: list):
        self.callbacks += callbacks

    def on_train_begin(self, logs=None):
        raise NotImplemented

    def on_train_end(self, logs=None):
        raise NotImplemented

    def on_epoch_begin(self, epoch, logs=None):
        raise NotImplemented

    def on_epoch_end(self, epoch, logs=None):
        raise NotImplemented

    def on_batch_begin(self, batch, logs=None):
        raise NotImplemented

    def on_batch_end(self, batch, logs=None):
        raise NotImplemented


class ModelWrapper(object):
    """
    Base class for all models. Supposed to be implementation and framework independent.
    """

    model_name = "name_not_specified"
    dataset_name = "dataset_not_specified"
    _saved_model_name = "saved_model.h5"
    callback_manager_class = CallbackManager
    callback_manager: callback_manager_class

    def __init__(self, **kwargs):
        self.args = kwargs
        self.prefix = kwargs.get('prefix', "default")
        self.dataset = self.load_dataset()
        if len(self.dataset) == 4:
            self.x_train, self.y_train, self.x_test, self.y_test = self.dataset
        new_model = kwargs.get("new_model", False)
        if new_model:
            self.model = self.construct()
            print("Constructed model %s" % self)
            self.summary()
        else:
            self.model = self.load()
            print("Loaded model %s" % self)

        self._elements = {}
        self._layers = []

        callbacks = kwargs.get("callbacks", [])

        self.callback_manager = self.callback_manager_class(callbacks)
        self.callback_manager.set_model(self.model)

    def put_group(self, group, elements, experiment=None):
        """
        Save a group of data associated with this model.
        :param experiment:
        :param group:
        :param elements:
        :return:
        """
        if group not in self._elements.keys():
            if experiment:
                if experiment not in self._elements[group].keys():
                    self._elements[group] = {}
                else:
                    self._elements[group][experiment].update(elements)
                self._elements[group][experiment] = elements
            else:
                self._elements[group] = elements
        else:
            if experiment:
                if type(self._elements[group]) != dict:
                    a = {"default": self._elements[group]}
                    self._elements[group] = a
                elif experiment not in self._elements[group].keys():
                    self._elements[group][experiment] = elements
                else:
                    self._elements[group][experiment].update(elements)
            else:
                self._elements[group].update(elements)
        return co.store_elements(group_name=group, model_wrapper=self, elements=elements, experiment=experiment)

    def get_group(self, group, experiment=None):
        """
        Load a group of data associated with this model.
        :param experiment:
        :param group:
        :return: the elements in the group.
        """

        if group not in self._elements.keys():
            data_store = co.load_elements(group_name=group, model_wrapper=self, experiment=experiment)
            if experiment:
                if experiment not in self._elements[group].keys():
                    self._elements[group] = {}
                self._elements[group][experiment] = data_store
            else:
                self._elements[group] = data_store

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
            return os.path.join(odin.results_dir, self.model_name, self.prefix)
        else:
            return os.path.join(odin.results_dir, self.model_name)

    @property
    def saved_model_path(self):
        return os.path.join(self.model_path, self._saved_model_name)

    @abstractmethod
    def construct(self):
        raise NotImplemented

    @abstractmethod
    def train(self, x_train=None, y_train=None, **options):
        raise NotImplemented

    @abstractmethod
    def save(self):
        raise NotImplemented

    @abstractmethod
    def layers(self, force_update=False) -> [LayerWrapper]:
        raise NotImplemented

    @abstractmethod
    def get_layer_outputs(self, x):
        raise NotImplemented

    @abstractmethod
    def weights(self):
        raise NotImplemented

    @abstractmethod
    def summary(self):
        raise NotImplemented

    def __str__(self):
        return "%s(%s):%s {dataset: %s} at '%s'" % (
            self.model_name, self.__class__.__name__, self.prefix, self.dataset_name, self.model_path)


available_models = {
    "keras_xor": "odin.models.keras_xor.KerasXOR",
    "mnist_vgg2": "odin.models.mnist_vgg2.MNISTWrapper",
    "cifar10_cnn": "odin.models.cifar10_cnn.Cifar10CNN",
    "rnn_lm": "odin.models.rnn_lm.RNNForLMWrapper",
    "mini_model": "odin.models.mini_model.MiniModel",
    "cifar10_wgan": "odin.models.wgan.keras_models.Cifar10WGAN",
    "mnist_wgan": "odin.models.wgan.keras_models.MnistWGAN",
    "info_gan": "odin.models.info_gan.trainer.InfoGAN",
    "pix2pix": "odin.models.pix2pix.Pix2Pix",
}


def load_model(model_name, **kwargs):
    model_name = model_name.lower()

    class_name = available_models.get(model_name)

    if not class_name:
        raise Exception("Unknown model %s" % model_name)

    class_obj = dynamic_class_import(class_name)

    assert issubclass(class_obj, ModelWrapper)

    model_wrapper = class_obj(**kwargs)
    odin.model_wrapper = model_wrapper

    return model_wrapper
