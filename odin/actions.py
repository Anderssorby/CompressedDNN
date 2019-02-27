import logging
from progress.bar import ChargingBar

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


def calc_index_set(**kwargs):
    """
    [WIP]
    Calculate the new architecture given already computed layer widths.

    Dependency: calc_dof

    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)

    method = "Newton-CG"
    r_dof = model_wrapper.get_group("range_test")
    # bounds = r_dof["rho_range"]
    layer_widths_list = r_dof["all_layer_widths"]
    # lambdas = r_dof["all_lambdas"]

    data = model_wrapper.get_group("inter_layer_covariance")
    cov_list = data["cov"]
    # eigen_values = data["eigen_values"]

    loss_before = []
    loss_after = []
    num = kwargs.get("num") or 1

    layer_widths = layer_widths_list[num]  # TODO this is not what we want
    logging.info("Finding the new architecture for %s" % layer_widths)
    print("Finding the new architecture for %s" % layer_widths)

    architecture.calculate_index_set(model_wrapper=model_wrapper, layer_widths=layer_widths,
                                     cov_list=cov_list)


def create_compressed_network(**kwargs):
    """
    Dependecy: calc_index_set
    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)

    r_dof = model_wrapper.get_group("range_test")
    layer_widths_list = r_dof["all_layer_widths"]

    data = model_wrapper.get_group("inter_layer_covariance")
    cov_list = data["cov"]
    model = architecture.transfer_to_architecture(model_wrapper=model_wrapper,
                                                  cov_list=cov_list)
    model.save()


def measure_goodness(**kwargs):
    """
    [WIP]

    Test the compressed model against the original model.

    :param kwargs:
    :return:
    """
    original_model = load_model(kwargs.get("model"), **kwargs)
    kwargs['prefix'] = kwargs['prefix'] + "_compressed"
    compressed_model = load_model(kwargs.get("model"), **kwargs)

    loss_original = original_model.test()

    loss_compressed = compressed_model.test()

    print(loss_original, loss_compressed)

    # Fine tune
    print("Fine tuning")
    compressed_model.train()

    loss_compressed_fine_tuned = compressed_model.test()
    print(loss_compressed_fine_tuned)

    compressed_model.prefix += "_fine_tuned"
    compressed_model.save()


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
    co.store_elements(model_wrapper=model_wrapper, experiment=kwargs.get("experiment"),
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
    Test LambdaOptimizer over a range
    Dependency: calc_eigs

    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)

    l_opt = lambda_param.LambdaOptimizer(model_wrapper)

    method = "Newton-CG"
    num = kwargs.get("num") or 1000
    max_rho = kwargs.get("max") or 0.15

    regularizing = kwargs.get("regularizing")

    rho_range = co.xp.linspace(0.01, max_rho, num=num)
    all_lambdas = []
    all_layer_widths = []
    bar = ChargingBar("Testing a range of rho on LambdaOptimizer", max=num)
    for rho in rho_range:
        lambdas, layer_widths = l_opt.optimize(rho=rho, method=method, debug=kwargs.get('verbose'),
                                               regularizing=regularizing)
        if kwargs['verbose']:
            print("rho=", rho, layer_widths, lambdas)
        all_lambdas.append(lambdas)
        all_layer_widths.append(layer_widths)
        bar.next()
    bar.finish()

    co.store_elements(model_wrapper=model_wrapper, group_name="range_test", experiment=kwargs.get("experiment"),
                      elements={"rho_range": rho_range, "all_lambdas": all_lambdas,
                                "all_layer_widths": all_layer_widths})
    if kwargs.get('plot'):
        range_test_plot(**kwargs)


def range_test_plot(**kwargs):
    """
    Plot the result of range test
    dependency: range_test

    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)
    data = model_wrapper.get_group("range_test", experiment=kwargs.get("experiment"))
    rho_range = data["rho_range"]
    all_lambdas = data["all_lambdas"]
    all_layer_widths = data["all_layer_widths"]

    n_layers = len(all_lambdas)

    fig = oplt.figure()
    # oplt.suptitle(r"Different layer widths for $\rho \in [0.01,0.15]$")
    oplt.subplot(2, 1, 1)
    oplt.plot(rho_range, all_lambdas)
    oplt.labels(r"$\rho$", "$\lambda_{\ell}$")
    oplt.legend([r"$\lambda_{%d}$" % l for l in range(n_layers)])
    oplt.subplot(2, 1, 2)
    oplt.plot(rho_range, all_layer_widths)
    oplt.labels(r"$\rho$", "$\hat{m}_{\ell}$")
    oplt.legend([r"$\hat{m}_{%d}$" % l for l in range(n_layers)])
    oplt.save("range_test", experiment=kwargs.get("experiment"))
    oplt.show()

    return fig


def calc_eigs(**kwargs):
    """
    Calculates the inter layer covariance and corresponding eigen values and stores them as 'inter_layer_covariance'.
    Dependency: train_model

    :param kwargs:
    :return:
    """
    model_wrapper = odin.model_wrapper = load_model(kwargs.get("model"), **kwargs)

    co.calc_inter_layer_covariance(model_wrapper=model_wrapper, **kwargs)

    datastore = model_wrapper.get_group("inter_layer_covariance")
    cov = datastore["cov"]
    eigen_values = datastore["eigen_values"]

    if kwargs["plot"]:
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
    # Remove None values
    kwargs = {a: b for a, b in kwargs.items() if b is not None}
    # kwargs['new_model'] = True
    model_wrapper = load_model(kwargs.get("model"), **kwargs)
    
    model_wrapper.train(**kwargs)
    model_wrapper.save()

    if model_wrapper.history:
        model_wrapper.put_group("training_history", {"history": model_wrapper.history.history})
        if kwargs["plot"]:
            oplt.plot_model_history(model_wrapper)
            oplt.save("loss")
            oplt.show()

    return model_wrapper


def test_model(**kwargs):
    """
    Test the specified model

    :param kwargs:
    :return:
    """
    model_wrapper = load_model(kwargs.get("model"), **kwargs)
    scores = model_wrapper.test()
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    return scores


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
    "range_test_plot": range_test_plot,
    "plot_dof": plot_dof,
    "calc_dof": calc_dof,
    "calc_eigs": calc_eigs,
    "train_model": train_model,
    "test_model": test_model,
    "measure_goodness": measure_goodness,
    "update_architecture": calc_index_set,
    "up_arch": calc_index_set,
    "calc_index_set": calc_index_set,
    "create_compressed_network": create_compressed_network,
    "compress": create_compressed_network,
    "": test_lambda_optimizer
}
