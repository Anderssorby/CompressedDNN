import unittest
import odin


class TestModelLoading(unittest.TestCase):
    def test_model(self):
        model_wrapper = odin.models.load_model(model_name="mnist_vgg2")

        self.assertIsNotNone(model_wrapper.model)
        model_wrapper = odin.models.load_model(model_name="cifar10_cnn")

        self.assertIsNotNone(model_wrapper.model)

    def test_model_layers(self):
        model_wrapper = odin.models.load_model(model_name="cifar10_cnn")
        layers = model_wrapper.layers()

        for layer in layers:
            layer
