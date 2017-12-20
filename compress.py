import keras.backend as K
from keras.models import Model
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def compute_covariance_matrix(model, layer, batch):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=layer.output)
    output = intermediate_layer_model.predict(batch)
    flat_shape = output[0].flatten().shape[0]
    # TODO handle
    sigma = np.zeros((flat_shape, flat_shape))
    n = len(batch)
    for i in range(n):
        g = output[i].flatten()
        sigma += np.outer(g, g)
    sigma = 1 / (n - 1) * sigma
    return sigma


def generalization_error(lambs, n):
    layer_widths = []
    bias = np.square(np.sum(np.sqrt(lambs)))
    variance = 0
    for l in range(layer_widths - 1):
        variance += layer_widths[l] * layer_widths[l + 1]
    variance *= np.log(n) / n

    return bias + variance


def plot_eigen_values(eigen_values):
    plt.figure()
    plt.plot(eigen_values)
    plt.title("Eigen values (%d)" % len(eigen_values))
    plt.draw()


class LambdaOptimizer:
    def __init__(self, model, x_train):
        self.model = model
        self.eigen_values = []
        self.covariance_matrices = []
        self.num_layers = len(model.layers)
        self.initial_widths = np.empty(self.num_layers)

        for n, layer in enumerate(model.layers):
            if layer.__class__ != Dense:
                print("Skipping...", layer)
                continue
            sigma = compute_covariance_matrix(model, layer, x_train[:100])

            eigen_values, eigen_vectors = np.linalg.eig(sigma)
            eigen_values = eigen_values[eigen_values != 0]
            eigen_values.sort()

            self.initial_widths[n] = int(layer.output.shape[-1])
            self.covariance_matrices.append(sigma)
            self.eigen_values.append(eigen_values)

    def objective_function(self, lamb):
        last = self.degree_of_freedom(lamb=lamb[0], layer=0)
        s = 0
        for layer in range(1, self.num_layers):
            n_l = self.degree_of_freedom(lamb[layer], layer)
            s += n_l * last
            last = self.degree_of_freedom(lamb[layer - 1], layer - 1)

        return s

    def plot_degree_of_freedom(self):
        X = np.linspace(0, 10)
        Y = [self.degree_of_freedom(l, 0) for l in X]
        plt.plot(X, Y)
        plt.title("Degree of freedom N(\lambda)")

    def optimize(self):

        # Solve N(l) = iw by min |N(l) - iw|
        # TODO this is excessive
        iw = self.initial_widths
        dof = self.degree_of_freedom

        def fun(lambs):
            n_of_lamb = [dof(lamb, layer) for layer, lamb in enumerate(lambs)]
            return np.linalg.norm(n_of_lamb - iw)

        lamb0 = np.ones(self.num_layers)*10
        res = opt.minimize(fun, lamb0)
        initial_lamb = res.x

        result = opt.minimize(self.objective_function, initial_lamb, method='L-BFGS-B', )

        print(result)
        lambdas = result.x
        layer_widths = [self.degree_of_freedom(lamb, layer) for layer, lamb in enumerate(lambdas)]

        return layer_widths

    def degree_of_freedom(self, lamb, layer):
        return np.sum([my / (my + lamb) for my in self.eigen_values[layer]])


class WidthOptimizer:
    def __init__(self, theoretical_widths=None, **kwargs):
        self.plot = False
        self.theoretical_widths = theoretical_widths
        self.__dict__.update(**kwargs)

    def compress(self, model, alpha=0.01, method="greedy", x_train=None,
                 y_train=None):

        for n, layer in enumerate(self.model.layers):
            print("Compressing layer %d - %s" % (n, str(layer.__class__)))
            if layer.__class__ != Dense:
                print("Skipping...")
                continue
            weights, biases = layer.get_weights()
            if method == "greedy":
                neurons, excluded = self._greedy(sigma, alpha)

                print('Theoretical %f and compressed size %d/%d' %
                      (self.theoretical_widths[n], len(neurons), len(neurons) + len(excluded)))
                print('Compressed layer', neurons)

                weights[:, excluded] = 0
                biases[excluded] = 0
                # weights = weights[:, neurons]
                # biases = biases[neurons]
                # TODO W=A*W
                adapted_layer = Dense(len(neurons))
                # layer.output_shape = len(neurons)
                layer.set_weights([weights, biases])
                sigma = compute_covariance_matrix(model, layer, x_train[:100])

                # compute_eigen_values(sigma, plot=True)
            else:
                A = self._group_sparse(sigma)

                adjusted_weights = A.dot(weights)

    def _greedy(self, cov, alpha):
        constraint = []
        possible = np.arange(cov.shape[0])
        neurons = []
        steps = 0
        while len(neurons) < len(possible):
            best_choice = 1e100
            best_index = None
            for j in possible:
                if j in neurons:
                    continue

                n_p_j = neurons + [j]
                reduced_cov = cov[np.ix_(n_p_j, n_p_j)]
                residual = -np.trace(reduced_cov)

                if residual < best_choice:
                    best_choice = residual
                    best_index = j

            if best_index is not None:
                neurons.append(best_index)
            else:
                raise Exception("No greedy choice")

            model_difference = np.trace(cov) + best_choice
            constraint.append(model_difference)
            if model_difference <= alpha:
                print('Finished after %d - %f' % (steps, model_difference))
                break

            if steps % 10 == 0:
                print('Step %d - %f' % (steps, model_difference))
            steps += 1

        # if self.plot:
        #     plt.figure()
        #     plt.plot(constraint)
        #     plt.title("Model difference")
        #     plt.draw()

        neurons.sort()
        return neurons, list(set(possible).difference(neurons))

    def _group_sparse(self, cov):
        # min tr(ASA^T - 2AS) + lambda * sum(norm(A[:, j]))
        lmb = 0.9

        def objective_function(a):
            return np.trace(a.dot(cov).dot(a.T) - 2 * a.dot(cov)) \
                   + lmb * np.sum([np.linalg.norm(a[:, j]) for j in range(a.shape[0])])

        max_iterations = 10000
        etha = 0.1
        n = cov.shape[0]
        I = np.eye(n)
        # Initial
        A = np.eye(n)
        steps = 0
        while steps < max_iterations:

            loss = objective_function(A)

            A = A - etha * (cov.dot(A) - 2 * cov + lmb * I)

            if loss < 1e-4:
                break
            steps += 1

        return A
