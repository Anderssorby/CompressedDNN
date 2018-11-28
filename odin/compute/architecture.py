"""
Architecture contains functions to compute a new subset of nodes after compression.
"""
import multiprocessing
import time

import numpy as np
import numpy.linalg as LA
import logging

import odin
import odin.plot as oplt
from odin.compute import default_interface as co


def pickle_fix(arg):
    """
    Makes nested functions picklable
    """
    return pickle_fix.calc(arg)


def greedy(constraint, indexes, m_l, parallel=False):
    """
    Greedy selection of nodes
    """

    selected = np.array([])
    plot = False
    choices = np.array(indexes)
    for i in range(len(selected), m_l):
        print("i = %d" % i)
        start = time.time()

        def calc(node):
            return constraint(np.union1d(selected, node))

        if parallel:
            pickle_fix.calc = calc
            available_cores = odin.config.get("available_cores", 4)
            pool = multiprocessing.Pool(processes=available_cores)
            values = pool.map(pickle_fix, choices)
            pool.close()
        else:
            # values: [float]
            values = list(map(calc, choices))

        greedy_choice = choices[np.argmax(values)]

        if plot:
            values = np.sort(values)
            oplt.plot(values)
            oplt.show()
            # current_best = np.max(values)

        selected = np.union1d(selected, [greedy_choice])
        choices = np.setdiff1d(choices, [greedy_choice])
        logging.debug("selected = %s; choice = %s; time = %.5f" % (
            selected, greedy_choice, time.time() - start))

    return selected


def compute_index_set(layer, cov, m_l, shape, weights, using_jl=False):
    if using_jl:
        # Somehow broken
        print("Enter Julia")

        from julia import Main
        Main.include("odin/compute/architecture.jl")
        j = Main.compute_index_set(layer, m_l, None)

        print("Exit julia")
        return j

    indexes = np.arange(shape)

    tr = np.trace
    theta = 0.5
    W = weights
    # R_z = W.T.dot(np.linalg.pinv(W.dot(W.T))).dot(W)

    def obj(j):
        j = list(map(int, j))
        f = np.setdiff1d(indexes, j)
        n = len(f)
        sig_inv = LA.inv(cov[np.ix_(j, j)])

        I = np.eye(n)
        R_z = np.eye(n)  # Projection matrix
        ch = theta * I + (1 - theta) * R_z

        difference = tr(ch.dot(cov[np.ix_(f, j)]).dot(sig_inv).dot(cov[np.ix_(j, f)]))
        normalizer = tr(ch.dot(cov[np.ix_(f, f)]))
        return difference / normalizer

    j, score = greedy(obj, indexes, m_l, parallel=True)

    return j


def transfer_to_architecture(model_wrapper, layer_widths, cov_list):
    regularizer_w = 1
    layers = model_wrapper.layers()
    weights = []
    biases = []
    for l, (layer, m_l, cov) in enumerate(zip(layers, layer_widths, cov_list)):
        if layer.type == "fully_connected":
            shape = cov.shape[0]  # layer.out_size
            indexes = np.arange(shape)
            start_time = time.time()
            logging.info("Will Compute index set for layer %d up to size %d" % (l, m_l))
            j = compute_index_set(l, cov, m_l, shape, layer.weights, using_jl=False)
            co.store_elements(group_name="index_set", elements={"layer_%d" % l: j},
                              model_wrapper=model_wrapper)
            elapsed_time = time.time() - start_time
            logging.info("Computed index set for layer %d in %f" % (l, elapsed_time))

            f = np.setdiff1d(indexes, j)

            _I_w = np.eye(m_l) * regularizer_w
            conversion_matrix = cov[np.ix_(f, j)] / (cov[np.ix_(j, j)] + _I_w)
            new_weights = conversion_matrix.dot(layer.weights)
            weights.append(new_weights)
            biases.append(layer.biases)

    new_wrapper = model_wrapper.__class__(layer_widths=layer_widths, prefix=(model_wrapper.prefix + "_compressed"))

    return new_wrapper


def error_reporter(error):
    print(error)
