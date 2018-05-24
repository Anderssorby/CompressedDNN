import odin.plot as oplt
from odin.compute import lambda_param
from odin.utils.default import default_chainer

if __name__ == "__main__":
    co, args, model_wrapper = default_chainer()

    datastore = model_wrapper.get_group("inter_layer_covariance")
    cov = datastore["cov"]
    eigen_values = datastore["eigen_values"]

    lopt = lambda_param.LambdaOptimizer(model_wrapper)

    lambdas, layer_widths = lopt.optimize(eigen_values, alpha=0.1)
    print(lambdas, layer_widths)

    lb = 0

    for i, eigs in enumerate(eigen_values):
        l = len(eigs)
        eigs = abs(eigs)
        eigs = eigs[eigs > lb]
        oplt.plot_eigen_values(eigs, title="%s layer %d (%d)" % (model_wrapper.model_name, i, l))
        oplt.save("eigs(%d)" % i)

    # oplt.show()
