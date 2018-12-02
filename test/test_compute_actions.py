import odin
import unittest
from odin.compute.lambda_param import *


class TestComputeActions(unittest.TestCase):

    def test_calc_eigs(self):
        action = odin.actions.action_map['calc_eigs']
        cov, eigen_values = action(model="mnist_vgg2")
        self.assertIsNotNone(cov)
        self.assertIsNotNone(eigen_values)

    def test_lambda_opt(self):
        mini = odin.models.load_model("mini")
        lambda_optimizer = LambdaOptimizer(mini)


