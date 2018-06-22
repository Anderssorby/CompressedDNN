import numpy as np
import scipy.linalg as LA
import scipy.optimize as opt


def frobenius_norm(matrix):
    rows, cols = matrix.shape
    norm = 0
    for i in range(rows):
        for j in range(cols):
            norm += matrix[i][j] ** 2
    norm = np.sqrt(norm)
    return norm


class LowRankApproximation:

    def __init__(self, rank):
        self.norm = frobenius_norm
        self.rank = rank

    def truncate(self, target, k):
        eigen_values, eigen_vectors = LA.eigvals(target)
        for i in range(k):
            eigen_values[i] = 0

        result = eigen_vectors * eigen_values * eigen_vectors.T

        return result

    def minimize(self, target):
        norm = self.norm

        def objective_function(m):
            return norm(m - target)

        m0 = np.zeros(target.shape)
        result = opt.minimize(objective_function, m0)

        return result


def pca(x):
    xxt = x.dot(x.T)
    eigs, eig_vs = LA.eigs(xxt)

    t = x * eig_vs
