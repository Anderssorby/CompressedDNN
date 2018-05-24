# from keras.models import Model
import logging

import numpy as np

from odin.compute import default_interface as co

accepted_layers = ['Dense', 'Conv2D', 'Conv1D']


def generalization_error(lambs, n):
    layer_widths = []
    bias = np.square(np.sum(np.sqrt(lambs)))
    variance = 0
    for l in range(len(lambs) - 1):
        variance += layer_widths[l] * layer_widths[l + 1]
    variance *= np.log(n) / n

    return bias + variance


class WidthOptimizer:
    def __init__(self, theoretical_widths=None, **kwargs):
        self.plot = False
        self.theoretical_widths = theoretical_widths
        self.__dict__.update(**kwargs)

    def compress(self, model, alpha=0.01, method="greedy", x_train=None,
                 y_train=None):

        for n, layer in enumerate(model.layers):
            logging.info("Compressing layer %d - %s" % (n, str(layer.__class__)))
            if layer.__class__.__name__ not in accepted_layers:
                logging.info("Skipping...")
                continue
            weights, biases = layer.get_weights()
            sigma = co.compute_covariance_matrix(model=model, layer=layer, batch=x_train)
            if method == "greedy":
                neurons, excluded = self._greedy(sigma, alpha)

                logging.info('Theoretical %f and compressed size %d/%d' %
                             (self.theoretical_widths[n], len(neurons), len(neurons) + len(excluded)))
                logging.debug('Compressed layer %s' % str(neurons))

                weights[:, excluded] = 0
                biases[excluded] = 0
                # weights = weights[:, neurons]
                # biases = biases[neurons]
                # TODO W=A*W
                # adapted_layer = Dense(len(neurons))
                # layer.output_shape = len(neurons)
                layer.set_weights([weights, biases])
                sigma = co.compute_covariance_matrix(model, layer, x_train[:100])

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
                logging.debug('Finished after %d - %f' % (steps, model_difference))
                break

            if steps % 10 == 0:
                logging.debug('Step %d - %f' % (steps, model_difference))
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
