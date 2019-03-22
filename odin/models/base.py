import os
from abc import abstractmethod, ABC

import odin
from odin.callbacks import CallbackManager
from odin.compute import default_interface as co
from odin.utils import dynamic_class_import


# from keras.models import load_model
# import keras.backend as K


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


class ModelWrapper(ABC):
    """
    Base class for all models. Supposed to be implementation and framework independent.
    """

    model_name = "name_not_specified"
    dataset_name = "dataset_not_specified"
    _saved_model_name = "saved_model.h5"
    callback_manager: CallbackManager

    def __init__(self, **kwargs):
        self.args = kwargs
        self.prefix = kwargs.get('prefix', "default")
        self.verbose = kwargs.get("verbose", False)
        self.dataset = self.load_dataset()
        if len(self.dataset) == 4:
            self.x_train, self.y_train, self.x_test, self.y_test = self.dataset
        new_model = kwargs.get("new_model", False)
        if new_model:
            self.model = self.construct()
            print("Constructed model %s" % self)
            if self.verbose:
                self.summary()
        else:
            self.model = self.load()
            print("Loaded model %s" % self)

        self._elements = {}
        self._layers = []

        callbacks = kwargs.get("callbacks", [])

        self.callback_manager = CallbackManager(callbacks)
        self.callback_manager.set_model(self.model)
        self.callback_manager.set_model_wrapper(self)

        odin.check_or_create_dir(self.model_path)

    def find_file(self, file):
        """
        :raises: ValueError if file is not a file or in model_path
        :param file: filename
        :return: A verified filename
        """
        if not os.path.isfile(file):
            file = os.path.join(self.model_path, file)
            if not os.path.isfile(file):
                raise ValueError("File not found: %s" % file)

        return file

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
    def save(self, **kwargs):
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
