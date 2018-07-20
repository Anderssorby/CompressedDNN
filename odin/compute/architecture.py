import chainer
import numpy as np
import numpy.linalg as LA


def greedy(constraint, indexes, m_l):
    j = set()
    choices = indexes
    for i in range(m_l):
        greedy_choice = None
        current_best = np.inf
        for node in choices:
            c = constraint(j.union([node]))
            if c < current_best:
                greedy_choice = node
                current_best = c

        j += greedy_choice
        choices.remove(greedy_choice)

    return j, c


def compute_index_set(cov, m_l, shape):
    indexes = np.arange(shape)

    def gen_fun(theta, width):
        tr = np.trace
        I = np.eye(width)
        R_z = np.eye(width)

        def fun(j):
            j = list(j)
            f = np.setdiff1d(indexes, j)
            sig_inv = LA.inv(cov[np.ix_(j, j)])
            difference = tr((theta * I + (1 - theta) * R_z) * cov[np.ix_(f, j)] * sig_inv * cov[np.ix_(j, f)])
            normalizer = tr(theta * I + (1 - theta) * R_z * cov[np.ix_(f, f)])
            return difference / normalizer

        return fun

    obj = gen_fun(theta=0.5, width=m_l)

    j, score = greedy(obj, indexes, m_l)

    return j


def transfer_to_architecture(model_wrapper, layer_widths, cov_list):
    regularizer_w = 1
    layers = model_wrapper.layers()
    weights = []
    biases = []
    for layer, m_l, cov in zip(layers, layer_widths, cov_list):
        if type(layer) == chainer.links.connection.linear.Linear:

            shape = layer.out_size
            indexes = np.arange(shape)
            j = compute_index_set(cov, m_l, shape)
            f = np.setdiff1d(indexes, j)

            I_w = np.eye(m_l) * regularizer_w
            conversion_matrix = cov[f, j] / (cov[j, j] + I_w)
            new_weights = conversion_matrix.dot(layer.W)
            weights.append(new_weights)
            biases.append(layer.b)

    new_wrapper = model_wrapper.__class__(layer_widths=layer_widths)

    return new_wrapper
