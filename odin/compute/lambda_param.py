import logging

import numpy as np
import scipy.optimize as opt

import odin.plot as oplt


def plot_lambdas(lambdas, layer_widths, optimal, prefix="line"):
    n = len(lambdas[0])

    def l_m2(l): return np.sum(np.sqrt(l), axis=1)

    def l_1(l): return np.sum(np.abs(l), axis=1)

    def only_col(l, col): return np.transpose(l)[col]

    fig, ax1 = oplt.subplots()

    legends = []

    for i in range(n):

        reg_x = l_m2(lambdas)
        reg_y = only_col(layer_widths, i)

        ax1.plot(reg_x, reg_y, 'b-')
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('lambdas')
        ax1.tick_params('y', colors='b')
        legends.append("$m_%d$" % i)

        ax1.plot(np.repeat(optimal[i], layer_widths.shape[1]))
        legends.append("optimal $m_%d$" % i)

    oplt.title("Lambda DOF (%s)" % prefix)

    oplt.labels(x_label="$\sum_l\sqrt{\lambda_\ell}$", y_label="$m_\ell$")
    oplt.legend(legends)

    oplt.save("%s_lambda_dof" % prefix)
    oplt.show()


def generate_n_sphere_points(r, n, num=100):
    x = np.empty((n, num))
    x.fill(r)
    phi = np.linspace(0, 2*np.pi, num=num)
    for i in range(n-1):
        for j in range(i, n):
            if j == i:
                x[j] = x[j] * np.cos(phi)
            else:
                x[j] = x[j] * np.sin(phi)

    return x.T


class LambdaOptimizer:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.eigen_values = model_wrapper.get_element("inter_layer_covariance", "eigen_values")
        self.model = model_wrapper.model
        self.lambdas = None
        self.layer_widths = None
        if self.model_wrapper.model_type == "keras":
            self.num_layers = len(self.model.layers)
            self.initial_widths = np.zeros(self.num_layers)
        elif self.model_wrapper.model_type == "chainer":
            layers = self.model_wrapper.layers()
            self.initial_widths = np.array([l.out_size for l in layers])
            self.num_layers = len(self.model_wrapper.layers())

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
        oplt.plot(X, Y)
        oplt.title("Degree of freedom N(\lambda)")

    def optimize_initial_lambda(self):
        # Solve N(l) = iw by min |N(l) - iw|
        iw = self.initial_widths
        dof = self.degree_of_freedom
        n = self.num_layers

        def fun(lambs):
            n_of_lamb = [dof(lamb, layer) for layer, lamb in enumerate(lambs)]
            return np.linalg.norm(n_of_lamb - iw)

        lamb0 = np.ones(n) * 0.1
        res = opt.minimize(fun, lamb0)
        initial_lamb = res.x
        return initial_lamb

    def optimize(self, bound=1, debug=False):
        dof = self.degree_of_freedom
        n = self.num_layers

        def bounding(l):
            return bound - np.sum(np.sqrt(l))

        constraint = {
            "type": "ineq",
            "fun": bounding
        }

        def gth0_gen(i):
            return lambda x: x[i]

        zero_constraints = [{
            "type": "ineq",
            "fun": gth0_gen(i)
        } for i in range(n)]

        bounds = [(0, None) for _ in range(n)]

        if debug:
            steps = []

            def callback(xk):
                steps.append(xk)
        else:
            callback = None

        lamb0 = np.ones(n) * 0.1

        result = opt.minimize(self.objective_function, lamb0,
                              constraints=[constraint] + zero_constraints,
                              bounds=bounds,
                              method='SLSQP',
                              options={"disp": True},
                              callback=callback)
        if debug:
            steps = np.array(steps)
            outs = np.array([self.objective_function(lamb) for lamb in steps])
            print(steps, outs)
            oplt.plot_line(steps, outs, title="Convergence")

            print(result.message)
        logging.debug(str(result))
        lambdas = result.x
        layer_widths = [dof(lamb, layer) for layer, lamb in enumerate(lambdas)]
        layer_widths = np.int32(np.floor(layer_widths))

        self.lambdas = lambdas
        self.layer_widths = layer_widths
        return lambdas, layer_widths

    def calc_layer_widths(self, lambdas):
        layer_widths = np.array([self.degree_of_freedom(lamb, layer) for layer, lamb in enumerate(lambdas)])
        return layer_widths

    def n_sphere_lambda_dof(self):
        n = self.num_layers

        sphere = generate_n_sphere_points(0.1, n)
        layer_widths = np.array([self.calc_layer_widths(lambdas) for lambdas in sphere])

        return sphere, layer_widths

    def range_lambda_dof(self):
        n = self.num_layers
        ones = np.ones(n)
        line = np.linspace(0, 0.5, num=100)
        l_arr = np.array([ones * x for x in line])

        layer_widths = np.array([self.calc_layer_widths(lambdas) for lambdas in l_arr])

        return l_arr, layer_widths

    def degree_of_freedom(self, lamb, layer):
        return np.sum([my / (my + lamb) for my in self.eigen_values[layer]])

    def result_text(self):
        result = "model_name = %s\n" % self.model_wrapper.model_name + \
                 "initial_widths = %s\n" % self.initial_widths + \
                 "lambdas = %s\n" % str(self.lambdas) + \
                 "layer_widths = %s\n" % str(self.layer_widths)

        return result
