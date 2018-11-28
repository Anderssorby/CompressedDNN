import logging

import odin
from odin import plot as oplt
from odin.models import load_model
from odin.compute import default_interface as co
from odin.compute import lambda_param, compress, architecture


def test_lambda_optimizer(**kwargs):
    """


    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)
    datastore = model_wrapper.get_group("inter_layer_covariance")
    cov = datastore["cov"]
    eigen_values = datastore["eigen_values"]

    l_opt = lambda_param.LambdaOptimizer(model_wrapper)

    w_opt = compress.CovarianceOptimizer(model_wrapper)

    lambdas, layer_widths = l_opt.optimize(method="Newton-CG", rho=1)

    lambdas, layer_widths = l_opt.optimize(method="SLSQP", bound=80)
    result = l_opt.result_text()
    print(l_opt.current_result)
    logging.info(result)
    print(result)

    w_result = w_opt.compress()
    print(w_result)


def update_architecture(**kwargs):
    """
    [WIP]
    Calculate the new architecture given already computed layer widths.

    Dependency: calc_dof

    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)

    method = "Newton-CG"
    r_dof = model_wrapper.get_group("range_dof_%s" % method)
    bounds = r_dof["bounds"]
    layer_widths_list = r_dof["layer_widths"]
    lambdas = r_dof["lambdas"]

    datastore = model_wrapper.get_group("inter_layer_covariance")
    cov_list = datastore["cov"]
    eigen_values = datastore["eigen_values"]

    loss_before = []
    loss_after = []

    layer_widths = layer_widths_list[1]

    new_model = architecture.transfer_to_architecture(model_wrapper=model_wrapper, layer_widths=layer_widths,
                                                      cov_list=cov_list)

    new_model.save()


def measure_goodness(**kwargs):
    """
    [WIP]

    Test the compressed model against the original model.

    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)

    method = "Newton-CG"
    r_dof = model_wrapper.get_group("range_dof_%s" % method)
    bounds = r_dof["bounds"]
    layer_widths_list = r_dof["layer_widths"]
    lambdas = r_dof["lambdas"]

    datastore = model_wrapper.get_group("inter_layer_covariance")
    cov_list = datastore["cov"]
    eigen_values = datastore["eigen_values"]

    loss_before = []
    loss_after = []

    layer_widths = layer_widths_list[1]

    new_model = architecture.transfer_to_architecture(model_wrapper=model_wrapper, layer_widths=layer_widths,
                                                      cov_list=cov_list)

    before = new_model.test()
    loss_before.append(before)

    # Fine tune
    new_model.train()

    after = new_model.test()
    loss_after.append(after)


def calc_dof(**kwargs):
    """
    This computes the layer widths for a range of hyper parameters for the given model.

    Dependency: calc_eigs


    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)

    l_opt = lambda_param.LambdaOptimizer(model_wrapper)

    method = "Newton-CG"
    line = co.xp.linspace(5.0, 100.0, num=100)
    lambdas, layer_widths, success = l_opt.range_lambda_dof(line, method=method)
    co.store_elements(model_wrapper=model_wrapper,
                      elements={"lambdas": lambdas, "bounds": line, "layer_widths": layer_widths, "success": success},
                      group_name="range_dof_%s" % method)


def plot_dof(**kwargs):
    """
    Makes a plot of layer widths given different hyper parameters
    Dependency: calc_dof

    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)

    # lambdas, optimal = l_opt.optimize(bound=0.1, debug=True)

    # l_arr, layer_widths = l_opt.n_sphere_lambda_dof()
    # lambda_param.plot_lambdas(l_arr, layer_widths, optimal=optimal, prefix="sphere")
    method = "Newton-CG"
    r_dof = model_wrapper.get_group("range_dof_%s" % method)
    bounds = r_dof["bounds"]
    layer_widths = r_dof["layer_widths"]
    lambdas = r_dof["lambdas"]

    lambda_param.plot_lambdas(lambdas, bounds, layer_widths, prefix=method)


def range_test(**kwargs):
    """


    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)

    l_opt = lambda_param.LambdaOptimizer(model_wrapper)

    bounds = [0.1, 0.5, 1, 2, 5, 10]
    for bound in bounds:
        lambdas, layer_widths = l_opt.optimize(bound=bound)
        print("bound=", bound, layer_widths, lambdas)


def calc_eigs(**kwargs):
    """
    Calculates the inter layer covariance and corresponding eigen values and stores them as 'inter_layer_covariance'.
    Dependency: train_model

    :param kwargs:
    :return:
    """
    model_wrapper = odin.model_wrapper = load_model(kwargs.get("model"), **kwargs)

    co.calc_inter_layer_covariance(model_wrapper=model_wrapper)

    datastore = model_wrapper.get_group("inter_layer_covariance")
    cov = datastore["cov"]
    eigen_values = datastore["eigen_values"]

    for i, eigs in enumerate(eigen_values):
        l = len(eigs)
        eigs = abs(eigs)
        oplt.plot_eigen_values(eigs, title="%s layer %d (%d)" % (model_wrapper.model_name, i, l))
        oplt.save("eigs(%d)" % i)
        oplt.plot_matrix(cov[i], title="Covariance (%d)" % i)
        oplt.save("cov(%d)" % i)

    return cov, eigen_values


def train_model(**kwargs):
    """
    Train the specified model. Commandline arguments will be passed to the training function.
    :param kwargs:
    :return:
    """
    kwargs['new_model'] = True
    model_wrapper = load_model(kwargs.get("model"), **kwargs)

    model_wrapper.train(**kwargs)
    model_wrapper.save()


def test_model(**kwargs):
    """
    Test the specified model

    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)

    model_wrapper.test(**kwargs)


class ActionManager:
    """
    Can manage several consecutive independent actions simplifying the experimental process.
    """

    def __init__(self, action_list):
        """

        :param action_list: A list of names of the actions to be performed in order
        """
        self.action_list = list(action_list)

    def check_and_execute(self):
        for action in self.action_list:
            # test_action_completed(action)
            action_function = action_map.get(action)
            if getattr(action_function, "test_completed", False):
                action_function(self.args)


action_map = {
    "test_lambda_optimizer": test_lambda_optimizer,
    "range_test": range_test,
    "plot_dof": plot_dof,
    "calc_dof": calc_dof,
    "calc_eigs": calc_eigs,
    "train_model": train_model,
    "measure_goodness": measure_goodness,
    "update_architecture": update_architecture,
    "": test_lambda_optimizer
}
