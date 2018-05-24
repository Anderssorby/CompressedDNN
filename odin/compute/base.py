from abc import abstractmethod
import odin.compute


class ComputationInterface(object):
    # abstracting computation operations

    def __init__(self, args):
        self.args = args
        odin.compute.default_interface = self

    @abstractmethod
    def calc_inter_layer_covariance(self, model_wrapper):
        pass

    @abstractmethod
    def load_group(self, group_name, model_name):
        pass

    @abstractmethod
    def store_elements(self, elements, element_name, model_name):
        pass
