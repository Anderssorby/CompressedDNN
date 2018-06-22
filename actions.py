import logging

from odin import plot as oplt
from odin.compute import default_interface as co
from odin.compute import lambda_param, compress


def test(model_wrapper, **kwargs):
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


def measure_goodness(model_wrapper, **kwargs):
    method = "Newton-CG"
    r_dof = model_wrapper.get_group("range_dof_%s" % method)
    bounds = r_dof["bounds"]
    layer_widths = r_dof["layer_widths"]
    lambdas = r_dof["lambdas"]

    for lw in layer_widths:
        new_model = model_wrapper.transfer_to_architecture(lw)



def calc_dof(model_wrapper, **kwargs):
    l_opt = lambda_param.LambdaOptimizer(model_wrapper)

    method = "Newton-CG"
    line = co.xp.linspace(5.0, 100.0, num=100)
    lambdas, layer_widths, success = l_opt.range_lambda_dof(line, method=method)
    co.store_elements(model_wrapper=model_wrapper,
                      elements={"lambdas": lambdas, "bounds": line, "layer_widths": layer_widths, "success": success},
                      group_name="range_dof_%s" % method)


def plot_dof(model_wrapper, **kwargs):
    # lambdas, optimal = l_opt.optimize(bound=0.1, debug=True)

    # l_arr, layer_widths = l_opt.n_sphere_lambda_dof()
    # lambda_param.plot_lambdas(l_arr, layer_widths, optimal=optimal, prefix="sphere")
    method = "Newton-CG"
    r_dof = model_wrapper.get_group("range_dof_%s" % method)
    bounds = r_dof["bounds"]
    layer_widths = r_dof["layer_widths"]
    lambdas = r_dof["lambdas"]

    lambda_param.plot_lambdas(lambdas, bounds, layer_widths, prefix=method)


def range_test(model_wrapper, **kwargs):
    l_opt = lambda_param.LambdaOptimizer(model_wrapper)

    bounds = [0.1, 0.5, 1, 2, 5, 10]
    for bound in bounds:
        lambdas, layer_widths = l_opt.optimize(bound=bound)
        print("bound=", bound, layer_widths, lambdas)


def calc_eigs(model_wrapper, **kwargs):
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


def train_model(model_wrapper, args, **kwargs):
    model_wrapper.train(args=args)


actions = {
    "test": test,
    "range_test": range_test,
    "plot_dof": plot_dof,
    "calc_dof": calc_dof,
    "calc_eigs": calc_eigs,
    "train_model": train_model,
    "": test
}
