from abc import abstractmethod


saving_dir = "computed"


class ComputationInterface(object):
    # abstracting computation operations

    def __init__(self, args):
        self.args = args

    @abstractmethod
    def calc_eigs(self, model_wrapper, exp_type, epoch_stage=15):
        pass
