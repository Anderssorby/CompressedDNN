import logging

from odin.compute import lambda_param, compress
from odin.utils.default import default_chainer


def test():
    lambdas, layer_widths = l_opt.optimize(bound=0.1)
    result = l_opt.result_text()

    logging.info(result)
    print(result)

    w_result = w_opt.compress()
    print(w_result)


def plot_dof():
    lambdas, optimal = l_opt.optimize(bound=0.1, debug=True)

    l_arr, layer_widths = l_opt.n_sphere_lambda_dof()
    lambda_param.plot_lambdas(l_arr, layer_widths, optimal=optimal, prefix="sphere")

    l_arr, layer_widths = l_opt.range_lambda_dof()
    lambda_param.plot_lambdas(l_arr, layer_widths, optimal=optimal, prefix="line")


def range_test():
    bounds = [0.1, 0.5, 1, 2, 5, 10]
    for bound in bounds:
        lambdas, layer_widths = l_opt.optimize(bound=bound)
        print("bound=", bound, layer_widths, lambdas)


actions = {
    "test": test,
    "range_test": range_test,
    "plot_dof": plot_dof,
    "": test
}

if __name__ == "__main__":
    co, args, model_wrapper = default_chainer()

    datastore = model_wrapper.get_group("inter_layer_covariance")
    cov = datastore["cov"]
    eigen_values = datastore["eigen_values"]

    l_opt = lambda_param.LambdaOptimizer(model_wrapper)

    w_opt = compress.CovarianceOptimizer(model_wrapper)

    actions[args.action]()
