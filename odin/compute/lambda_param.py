import logging

import numpy as np
import scipy.optimize as opt

import odin.plot as oplt


def plot_lambdas(lambdas, bounds, layer_widths, prefix="line"):
    n = len(layer_widths[0])

    def l_m2(l): return np.sum(np.sqrt(l), axis=1)

    def l_1(l): return np.sum(np.abs(l), axis=1)

    def only_col(l, col): return np.transpose(l)[col]

    fig, (ax1, ax2) = oplt.subplots(2, 1, sharex=True)
    ax1.set_xlabel("$\sum_l\sqrt{\lambda_\ell}\leq D$")
    ax1.set_ylabel("$m_\ell$")

    color = 'tab:blue'
    ax2.set_ylabel('$\lambda_\ell$', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)

    # legends = []

    for i in range(n):
        reg_y = only_col(layer_widths, i)

        handle = ax1.plot(bounds, reg_y, '-', label="$m_%d$" % i)

        # legends.append(handle)

        # ax1.plot(np.repeat(optimal[i], layer_widths.shape[1]))
        # legends.append("optimal $m_%d$" % i)

        lamb_i = only_col(lambdas, i)
        handle = ax2.plot(bounds, lamb_i, '-', label='$\lambda_%d$' % i)
        # legends.append(handle)

    oplt.title("Optimization of DOF (%s)" % prefix)

    ax1.legend()
    ax2.legend()

    fig.tight_layout()

    oplt.save("%s_lambda_dof" % prefix)
    oplt.show()


def generate_n_sphere_points(r, n, num=100):
    x = np.empty((n, num))
    x.fill(r)
    phi = np.linspace(0, 2 * np.pi, num=num)
    for i in range(n - 1):
        for j in range(i, n):
            if j == i:
                x[j] = x[j] * np.cos(phi)
            else:
                x[j] = x[j] * np.sin(phi)

    return x.T


class LambdaOptimizer:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper

        eig_rows = model_wrapper.get_element("inter_layer_covariance", "eigen_values")
        # remove negative eigenvalues
        epsilon = 1
        for i, eigs in enumerate(eig_rows):
            eigs[eigs < epsilon] = epsilon
        eig_rows = np.abs(eig_rows)

        self.layer_eigen_values = eig_rows

        self.model = model_wrapper.model
        # self.lambdas = None
        # self.layer_widths = None
        self.current_result = None
        if self.model_wrapper.model_type == "keras":
            self.num_layers = len(self.model.layers()) - 1
            self.initial_widths = np.zeros(self.num_layers)
        elif self.model_wrapper.model_type == "chainer":
            layers = self.model_wrapper.layers()
            self.initial_widths = np.array([l.units for l in layers])
            self.num_layers = len(self.model_wrapper.layers()) - 1  # Do not include the last layer

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

    def _optimize_SLSQP(self, bound=1.0, debug=False):
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

        obj = self.objective_function
        rho = bound

        def lagr_obj(x):
            obj(x ** 2) + rho * np.sum(x)

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

        if not result.success:
            logging.info("Unsuccessful for bound=%.5f, constraint=%.5f", bound, bounding(lamb0))

        return result

    def _optimize_newton_cg(self, rho=1):
        obj = self.objective_function
        n = self.num_layers
        ones = np.ones(n)
        lamb0 = ones * 0.1

        def lagrangian(x):
            return obj(x ** 2) + rho * np.sum(x)

        jacobian = self.jacobian

        def jac(x):
            return 2 * x * jacobian(x ** 2) + rho * ones

        result = opt.minimize(lagrangian, lamb0,
                              jac=jac,
                              method='Newton-CG',
                              options={"disp": True})

        return result

    def optimize(self, method, **kwargs):
        dof = self.degree_of_freedom
        if method == "SLSQP":
            result = self._optimize_SLSQP(**kwargs)
            lambdas = result.x

        elif method == "Newton-CG":
            result = self._optimize_newton_cg(**kwargs)
            lambdas = np.sqrt(result.x)
        else:
            raise ValueError("Unkown method %s" % method)

        logging.debug(str(result))

        layer_widths = [dof(lamb, layer) for layer, lamb in enumerate(lambdas)]
        layer_widths = np.int32(np.floor(layer_widths))

        self.current_result = result
        if not result.success:
            logging.info("Unsuccessful")

        return lambdas, layer_widths

    def calc_layer_widths(self, lambdas):
        layer_widths = np.array([self.degree_of_freedom(lamb, layer) for layer, lamb in enumerate(lambdas)])
        return layer_widths

    def n_sphere_lambda_dof(self):
        n = self.num_layers

        sphere = generate_n_sphere_points(0.1, n)

        l_arr = []

        for bound in sphere:
            lambdas, _ = self.optimize(bound=bound, debug=False)
            l_arr.append(lambdas)

        layer_widths = np.array([self.calc_layer_widths(lambdas) for lambdas in sphere])

        return sphere, layer_widths

    def range_lambda_dof(self, line, method="SLSQP"):
        n = self.num_layers

        l_arr = []
        layer_widths = []
        success = []

        for bound in line:
            if method.upper() == "SLSQP":
                lambdas, m_ls = self.optimize(bound=bound, debug=False)
            elif method.upper() == "NEWTON-CG":
                lambdas, m_ls = self.optimize(method="Newton-CG", rho=bound)
            else:
                raise ValueError(method)

            l_arr.append(lambdas)
            layer_widths.append(m_ls)
            success.append(self.current_result.success)

        # layer_widths = np.array([self.calc_layer_widths(lambdas) for lambdas in l_arr])

        return l_arr, layer_widths, success

    def jacobian(self, lamb):
        n = self.num_layers
        mu = self.layer_eigen_values

        dof = self.degree_of_freedom

        def der_dof(lamb, layer):
            return -np.sum([mu_lk / (mu_lk + lamb) ** 2 for mu_lk in mu[layer]])

        jac = np.zeros(n)

        for l in range(0, n):
            n_der_l = der_dof(lamb[l], l)
            s = 0
            if l < n - 1:
                s += dof(lamb[l + 1], l) * n_der_l
            if l > 0:
                s += dof(lamb[l - 1], l) * n_der_l
            jac[l] = s

        return jac

    def degree_of_freedom(self, lamb, layer):
        return np.sum([my / (my + lamb) for my in self.layer_eigen_values[layer]])

    def result_text(self):
        lambdas = self.current_result.x
        result = "model_name = %s\n" % self.model_wrapper.model_name + \
                 "initial_widths = %s\n" % self.initial_widths + \
                 "lambdas = %s\n" % str(lambdas) + \
                 "layer_widths = %s\n" % str(self.calc_layer_widths(lambdas))

        return result
