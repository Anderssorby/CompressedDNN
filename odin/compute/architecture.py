"""
Architecture contains functions to compute a new subset of nodes after compression.
"""
import multiprocessing
import time

import numpy as np
import numpy.linalg as LA
import logging
from progress.bar import ChargingBar
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
    bar = ChargingBar("Calculating index set with greedy method", max=m_l)

    for i in range(len(selected), m_l):
        # print("i = %d" % i)
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
        bar.next()
    bar.finish()

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
        sig_inv = LA.pinv(cov[np.ix_(j, j)])

        I = np.eye(n)
        R_z = np.eye(n)  # Projection matrix
        ch = theta * I + (1 - theta) * R_z

        difference = tr(ch.dot(cov[np.ix_(f, j)]).dot(sig_inv).dot(cov[np.ix_(j, f)]))
        normalizer = tr(ch.dot(cov[np.ix_(f, f)]))
        return difference / normalizer

    j = greedy(obj, indexes, m_l, parallel=False)

    j = j.astype(dtype=int)

    return j


def calculate_index_set(model_wrapper, layer_widths, cov_list):
    layers = model_wrapper.layers()
    index_set = []
    for l, (layer, m_l, cov) in enumerate(zip(layers, layer_widths, cov_list)):
        if layer.type == "fully_connected":
            shape = cov.shape[0]  # layer.out_size
            start_time = time.time()
            logging.info("Will Compute index set for layer %d up to size %d" % (l, m_l))
            j = compute_index_set(l, cov, m_l, shape, layer.weights, using_jl=False)

            assert len(j) == m_l

            model_wrapper.put_group(group="index_set", elements={"layer_%d" % l: j})

            elapsed_time = time.time() - start_time
            logging.info("Computed index set for layer %d in %f" % (l, elapsed_time))

    return index_set


def transfer_to_architecture(model_wrapper, cov_list):
    regularizer_w = 1
    layers = model_wrapper.layers()
    weights = []
    biases = []
    index_sets = model_wrapper.get_group(group="index_set")
    layer_widths = []
    last_j = None

    for l, (layer, cov) in enumerate(zip(layers[0:-1], cov_list)):  # skip the last layer
        if layer.type == "fully_connected":
            shape = layer.units
            indexes = np.arange(shape)

            j = index_sets["layer_%d" % l]
            m_l = len(j)
            layer_widths.append(m_l)

            f = np.setdiff1d(indexes, j)

            _I_w = np.eye(m_l) * regularizer_w
            conversion_matrix = cov[np.ix_(f, j)].dot(LA.inv(cov[np.ix_(j, j)] + _I_w))
            if last_j is not None:
                sub_weights = layer.weights[np.ix_(last_j, j)]
            else:
                sub_weights = layer.weights[:, j]

            new_weights = sub_weights  # conversion_matrix.dot()
            weights.append(new_weights)
            if layer.biases:
                biases.append(layer.biases[j])

            last_j = j

    last_layer = layers[-1]
    sub_weights = last_layer.weights[last_j, :]
    new_weights = sub_weights  # conversion_matrix.dot()
    weights.append(new_weights)

    new_wrapper = model_wrapper.__class__(layer_widths=layer_widths, weights=weights, biases=biases,
                                          prefix=(model_wrapper.prefix + "_compressed"), new_model=True)

    return new_wrapper


def error_reporter(error):
    print(error)
