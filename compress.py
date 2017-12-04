import keras.backend as K
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

class WidthOptimizer:

    def __init__(self, model,**kwargs):
        self.model = model

        self.nth_layer_function = lambda n: K.function([model.layers[0].input],
                                  [model.layers[n].output])
        self.__dict__.update(**kwargs)

    def compute_covariance_matrix(self, layer):
        intermediate_layer_model = Model(inputs=self.model.input,
                                    outputs=layer.output)
        batch = self.x_train[:100]
        output = intermediate_layer_model.predict(batch)
        sigma = np.zeros((output[0].shape[0], output[0].shape[0]))
        # print(output.shape)
        n = len(batch)
        for i in range(n):
            sigma += np.outer(output[i], output[i])
        sigma = 1/(n-1) * sigma
        return sigma

    def plot_eigen(self, eigen_values):
        print(eigen_values)

        plt.loglog(eigen_values)
        plt.title("Eigen values")
        plt.show()

    def plot_degree_of_freedom(self):
        X = np.linspace(0, 10)
        Y = [self.degree_of_freedom(l) for l in X]
        plt.plot(X, Y)
        plt.title("Degree of freedom N(\lambda)")
        plt.show()

    def compress(self, method="greedy"):
        # Compress each layer
        for n, layer in enumerate(self.model.layers):
            print("Compressing layer %d - %s" % (n, str(layer.__class__)))
            weights, biases = layer.get_weights() # list of numpy arrays

            sigma = self.compute_covariance_matrix(layer)

            eigen_values, eigen_vectors = np.linalg.eig(sigma)
            eigen_values.sort()

            self.degree_of_freedom = lambda l: sum([my/(my+l) for my in eigen_values])

            # self.plot_eigen(eigen_values)

            l = 0.6
            m_l = self.degree_of_freedom(l)
            if method == "greedy":
                neurons = self._greedy(sigma)

                print('Degree of freedom %f and compressed size %d' % (m_l, len(neurons)))
                print('Compressed layer', neurons)

                # Update weights
                adjusted_weights = weights[:, neurons]
                adjusted_biases = biases[neurons]
                # TODO W=A*W
                # compressed_layer =
                layer.set_weights([adjusted_weights, adjusted_biases])
            else:
                A = self._group_sparse(sigma)

                adjusted_weights = A.dot(weights)

    def _greedy(self, cov):
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
            if model_difference <= self.alpha:
                print('Finished after %d - %f' % (steps, model_difference))
                break

            if steps % 10 == 0:
                print('Step %d - %f' % (steps, model_difference))
            steps += 1

        plt.plot(constraint)
        plt.title("Model difference")
        plt.show()
        neurons.sort()
        return neurons

    def _group_sparse(self, cov, weights):
        # min tr(ASA^T - 2AS) + lambda * sum(norm(A[:, j]))
        objective_function = lambda A: np.trace(A.dot(cov).dot(A.T) - 2*A.dot(cov)) \
            + self.lmb * np.sum([np.linalg.norm(A[:, j]) for j in range(A.shape[0])])

        max_iterations = 10000
        etha = 0.1
        n = cov.shape[0]
        I = np.eye(n)
        # Initial
        A = np.eye(n)
        steps = 0
        while steps < max_iterations:

            loss = objective_function(A)

            A = A - etha*(cov.dot(A) - 2*cov + lmb*I)

            if loss < 1e-4:
                break
            steps += 1

        return A

    def generalization_error(self, lmb):

        layer_widths = []
        lmbs = []
        bias = np.square(np.sum(np.sqrt(lmbs)))
        variance = 0
        for l in range(layer_widths-1):
            variance += layer_widths[l]*layer_widths[l+1]
        variance *= np.log(n)/n

        return bias + variance



