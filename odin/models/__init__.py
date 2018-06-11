
def load_model(model_name, **kwargs):
    model_name = model_name.lower()

    # module = __import__("odin.models.%s" % model_name)
    # cls = getattr(module, "Wrapper")
    if model_name == "keras_xor":
        from odin.models.keras_xor import KerasXOR
        model_wrapper = KerasXOR(**kwargs)
    elif model_name == "vgg2":
        from odin.models.vgg2 import VGG2Wrapper
        model_wrapper = VGG2Wrapper(**kwargs)
    elif model_name == "mnist_vgg2":
        from odin.models.mnist_vgg2 import MNISTWrapper
        model_wrapper = MNISTWrapper(**kwargs)
    elif model_name == "rnn_lm":
        from odin.models.rnn_lm import RNNForLMWrapper
        model_wrapper = RNNForLMWrapper(**kwargs)
    elif model_name == "cifar10_wgan":
        from odin.models.wgan import WGANWrapper
        model_wrapper = WGANWrapper(**kwargs)
    # elif model_name == "mnist_cnn":
    #    from models.mnist_cnn import model, x_train, y_train, x_test, y_test
    # elif model_name == "cifar10_cnn":
    #     from models.cifar10_cnn import model, x_train, y_train, x_test, y_test
    else:
        raise Exception("Unknown model %s" % model_name)
    # model.model_name = model_name
    # return model, x_train, y_train
    return model_wrapper
