from abc import abstractmethod


class ComputationInterface(object):
    # abstracting computation operations

    def __init__(self):
        self.args = None

    def update_args(self, args):
        self.args = args

    @abstractmethod
    def calc_inter_layer_covariance(self, model_wrapper):
        pass

    @abstractmethod
    def load_group(self, group_name, model_name):
        pass

    @abstractmethod
    def store_elements(self, elements, element_name, model_name):
        pass
