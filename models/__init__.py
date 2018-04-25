
def load_model(model_name):
    if model_name == "keras_xor":
        from models.keras_xor import KerasXOR
        model_wrapper = KerasXOR()
        return model_wrapper.model, model_wrapper.x_train, model_wrapper.y_train
    elif model_name == "mnist_cnn":
        from models.mnist_cnn import model, x_train, y_train, x_test, y_test
    elif model_name == "cifar10_cnn":
        from models.cifar10_cnn import model, x_train, y_train, x_test, y_test
    else:
        raise Exception("Unknown model %s" % model_name)
    model.model_name = model_name
    return model, x_train, y_train
