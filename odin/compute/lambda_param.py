import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


class LambdaOptimizer:
    def __init__(self, model_wrapper):
        self.mw = model_wrapper
        self.eigen_values = model_wrapper.get_element("inter_layer_covariance", "cov")
        self.model = model_wrapper.model
        self.eigen_values = []
        self.covariance_matrices = []
        if self.mw.model_type == "keras":
            self.num_layers = len(self.model.layers)
        elif self.mw.model_type == "chainer":
            layers = self.mw.layers()
            self.initial_widths = np.array([l.out_size for l in layers])
            self.num_layers = len(self.mw.layers())
        self.initial_widths = np.zeros(self.num_layers)

    def objective_function(self, lamb):
        last = self.degree_of_freedom(lamb=lamb[0], layer=0)
        s = 0
        for layer in range(1, self.num_layers):
            n_l = self.degree_of_freedom(lamb[layer], layer)
            s += n_l * last
            last = n_l

        return s

    def plot_degree_of_freedom(self):
        X = np.linspace(0, 10)
        Y = [self.degree_of_freedom(l, 0) for l in X]
        plt.plot(X, Y)
        plt.title("Degree of freedom N(\lambda)")

    def optimize(self, eigen_values, alpha=1):
        self.eigen_values = eigen_values

        # Solve N(l) = iw by min |N(l) - iw|
        # TODO this is excessive
        iw = self.initial_widths
        dof = self.degree_of_freedom
        n = self.num_layers

        def fun(lambs):
            n_of_lamb = [dof(lamb, layer) for layer, lamb in enumerate(lambs)]
            return np.linalg.norm(n_of_lamb - iw)

        lamb0 = np.ones(n) * 0.1
        res = opt.minimize(fun, lamb0)
        initial_lamb = res.x

        def bounding(l):
            return alpha - np.sum(l)

        constraint = {
            "type": "ineq",
            "fun": bounding
        }

        def gth0(x):
            return x

        # zero_constraints = [{
        #     "type": "ineq",
        #     "fun": gth0
        # } for _ in range(n)]
        #
        bounds = [(0, None) for _ in range(self.num_layers)]

        result = opt.minimize(self.objective_function, lamb0,
                              constraints=[constraint],
                              bounds=bounds,
                              method='trust-constr',
                              options={"disp": True})

        print(result.message)
        logging.debug(str(result))
        lambdas = result.x
        layer_widths = [dof(lamb, layer) for layer, lamb in enumerate(lambdas)]

        return lambdas, layer_widths

    def degree_of_freedom(self, lamb, layer):
        return np.sum([my / (my + lamb) for my in self.eigen_values[layer]])
