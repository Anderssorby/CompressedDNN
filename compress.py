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
            sigma = np.zeros(batch[0].shape)
            output = intermediate_layer_model.predict(batch)
            print(output.shape)
            n = len(batch)
            for i in range(n):
                import pdb; pdb.set_trace()  # XXX BREAKPOINT
                sigma += np.outer(output[i], output[i])
            sigma = 1/(n-1) * sigma
            return sigma

    def compress(self, method="greedy"):
        # Compress each layer
        for n, layer in enumerate(self.model.layers):
            print("Compressing layer %d" % n)
            weights = layer.get_weights() # list of numpy arrays

            sigma = self.compute_covariance_matrix(layer)

            eigen_values, eigen_vectors = np.linalg.eig(sigma)
            eigen_values.sort()
            # print(eigen_values)
            # plt.loglog(eigen_values)
            # plt.show()
            degree_of_freedom = lambda l: sum([my/(my+l) for my in eigen_values])
            l = 0.6
            m_l = degree_of_freedom(l)

            if method == "greedy":
                neurons = self._greedy(sigma)
                print(f'Degree of freedom {m_l} and compressed size {len(neurons)}')
                print(f'Compressed layer {neurons}')

                # Update weights
                adjusted_weights = weights[neurons]
                # TODO W=A*W
                layer.set_weights(adjusted_weights)
            else:
                A = self._group_sparse(sigma)

                adjusted_weights = weights @ A

    def _greedy(self, cov):
        constraint = []
        # print(cov.shape[0])
        possible = np.arange(cov.shape[0])
        neurons = []
        steps = 0
        while len(neurons) < len(possible):
            # Make greedy choice
            best_choice = 1e100
            best_index = None
            for j in possible:
                if j in neurons:
                    continue
                reduced_cov = cov[neurons]
                # print(reduced_cov)
                residual = -np.trace(reduced_cov)
                # print(residual)
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
                print(f'Finished after {steps} - {model_difference}')
                break

            if steps % 10 == 0:
                print(f'{steps} - {model_difference}')
            steps += 1

        plt.plot(constraint)
        plt.title("Model difference")
        plt.show()

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



