"""
Architecture contains functions to compute a new subset of nodes after compression.
"""
import multiprocessing
import time

import numpy as np
import numpy.linalg as LA
import chainer

import odin.plot as oplt


def pickle_fix(arg):
    """
    Makes nested functions pickable
    """
    pickle_fix.calc(arg)


def greedy(constraint, indexes, m_l):
    """
    Greedy selection of nodes
    """
    already = [855, 848, 625, 302, 948, 141, 16, 364, 801, 179,
               843, 690, 540, 608, 705, 363, 878, 720, 40, 436,
               780, 365, 560, 825, 571, 404]

    selected = np.array(already, dtype=np.int16)  # []
    plot = False
    parallel = True
    choices = indexes
    for i in range(len(selected), m_l):
        print("i = %d" % i)
        start = time.time()

        def calc(node):
            return constraint(np.union1d(selected, node))

        if parallel:
            pickle_fix.calc = calc
            pool = multiprocessing.Pool(processes=4)
            values = pool.map_async(pickle_fix, choices, error_callback=error_reporter)
            pool.close()
        else:
            values: [float]
            values = map(calc, choices)

        greedy_choice = np.argmax(values)

        if plot:
            values.sort()
            oplt.plot(values)
            oplt.show()
            # current_best = np.max(values)

        selected = np.union1d(selected, [greedy_choice])
        choices = np.setdiff1d(choices, [greedy_choice])
        print("selected = %s; choice = %s; time = %.5f" % (
            selected, greedy_choice, time.time() - start))

    return selected


def compute_index_set(cov, m_l, shape, weights):
    indexes = np.arange(shape)

    tr = np.trace
    theta = 0.5
    W = weights
    # R_z = W.T.dot(np.linalg.pinv(W.dot(W.T))).dot(W)

    def obj(j):
        f = np.setdiff1d(indexes, j)
        n = len(f)
        sig_inv = LA.inv(cov[np.ix_(j, j)])

        I = np.eye(n)
        R_z = np.eye(n)  # Projection matrix
        ch = theta * I + (1 - theta) * R_z

        difference = tr(ch.dot(cov[np.ix_(f, j)]).dot(sig_inv).dot(cov[np.ix_(j, f)]))
        normalizer = tr(ch.dot(cov[np.ix_(f, f)]))
        return difference / normalizer

    # t = time.time
    # s = t()
    # obj([1, 10, 100])
    # e = t()-s
    # print("time_of(obj)=%f" % e)

    j, score = greedy(obj, indexes, m_l)

    return j


def transfer_to_architecture(model_wrapper, layer_widths, cov_list):
    regularizer_w = 1
    layers = model_wrapper.layers()
    weights = []
    biases = []
    for layer, m_l, cov in zip(layers, layer_widths, cov_list):
        if type(layer) == chainer.links.connection.linear.Linear:
            shape = cov.shape[0]  # layer.out_size
            indexes = np.arange(shape)
            j = compute_index_set(cov, m_l, shape, layer.W)
            f = np.setdiff1d(indexes, j)

            _I_w = np.eye(m_l) * regularizer_w
            conversion_matrix = cov[np.ix_(f, j)] / (cov[np.ix_(j, j)] + _I_w)
            new_weights = conversion_matrix.dot(layer.W)
            weights.append(new_weights)
            biases.append(layer.b)

    new_wrapper = model_wrapper.__class__(layer_widths=layer_widths, prefix=str(time.time()) + "_transferred_")

    return new_wrapper


def error_reporter(error):
    print(error)
