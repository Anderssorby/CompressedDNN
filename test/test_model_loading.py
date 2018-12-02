import unittest
import odin
import odin.models.base


class TestModelLoading(unittest.TestCase):
    def test_model(self):
        model_wrapper = odin.models.base.load_model(model_name="mnist_vgg2")

        self.assertIsNotNone(model_wrapper.model)
        model_wrapper = odin.models.base.load_model(model_name="cifar10_cnn")

        self.assertIsNotNone(model_wrapper.model)

    def test_model_layers(self):
        model_wrapper = odin.models.base.load_model(model_name="cifar10_cnn")
        layers = model_wrapper.layers()

        for layer in layers:
            print(layer)

        (x, y), _ = model_wrapper.dataset
        print(x.shape)

        output = model_wrapper.get_layer_outputs(x[0:100])
        print(output)

    def test_mini(self):
        mini = odin.models.load_model("mini", new_model=True)
        mini.train()
