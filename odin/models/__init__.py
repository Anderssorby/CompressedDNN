from odin.utils import dynamic_class_import
from .base import ModelWrapper

available_models = {
    "keras_xor": "odin.models.keras_xor.KerasXOR",
    "cifar10_wgan": "odin.models.mnist_vgg2.MNISTWrapper",
    "mnist_vgg2": "odin.models.mnist_vgg2.MNISTWrapper",
    "cifar10_cnn": "odin.models.cifar10_cnn.Cifar10CNN",
    "rnn_lm": "odin.models.rnn_lm.RNNForLMWrapper",
}


def load_model(model_name, **kwargs):
    model_name = model_name.lower()

    class_name = available_models.get(model_name)

    if not class_name:
        raise Exception("Unknown model %s" % model_name)

    class_obj = dynamic_class_import(class_name)

    assert issubclass(class_obj, ModelWrapper)

    model_wrapper = class_obj(**kwargs)

    return model_wrapper
