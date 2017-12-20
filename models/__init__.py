
def load_model(model_name):
    if model_name == "keras_xor":
        from models.keras_xor import model, training_data, target_data
        return model, training_data, target_data

    elif model_name == "mnist_cnn":
        from models.mnist_cnn import model, x_train, y_train, x_test, y_test
        return model, x_train[:1000], y_train[:1000]
    else:
        raise Exception(f"Unknown model {model_name}")
