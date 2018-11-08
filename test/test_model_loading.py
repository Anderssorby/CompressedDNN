import unittest
import odin


class TestModelLoading(unittest.TestCase):
    def test_model(self):
        model_wrapper = odin.models.load_model(model_name="mnist_vgg2")

        self.assertIsNotNone(model_wrapper.model)
