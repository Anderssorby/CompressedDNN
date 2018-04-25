import argparse
import logging
import os
import time

import keras
import matplotlib.pyplot as plt
import numpy as np

from compute.compress import WidthOptimizer, LambdaOptimizer
from models import load_model


def prepare_logging(args):
    log_dir = os.path.join('log', str(os.path.basename(args['model']).split('.')[0]),
                           time.strftime('%Y-%m-%d_%H-%M-%S_') + str(time.time()).replace('.', ''))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'log.txt')
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_file, level=logging.DEBUG)
    logging.info(args)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False,
                    default="keras_xor", help="model to compress")
    ap.add_argument("-f", "--file-prefix", required=False,
                    default="save", help="prefix for the files of computed values")
    ap.add_argument('--seed', type=int, default=1701)
    args = vars(ap.parse_args())
    prepare_logging(args)

    np.random.seed(args['seed'])

    model, training_data, target_data = load_model(args["model"])

    logging.info("Optimizing theoretical layer width")
    lambda_optimizer = LambdaOptimizer(model=model)
    lambda_optimizer.compute_or_load(file_prefix=args["file_prefix"], x_train=training_data)
    lambdas = lambda_optimizer.optimize()
    logging.debug("Lambdas = %s" % str(lambdas))

    n = 10
    alphas = np.linspace(0.01, 0.1, num=n)

    initial_loss, v = model.test_on_batch(training_data, target_data)
    logging.info("Test before compression %s" % str(initial_loss))

    compressed_loss = np.zeros(n)
    fine_tuned_loss = np.zeros(n)

    for i, alpha in enumerate(alphas):
        logging.info("alpha = %s" % str(alpha))

        optimizer = WidthOptimizer(theoretical_widths=lambdas, lmb=0.8, plot=True)

        compressed_model = keras.models.clone_model(model)
        optimizer.compress(compressed_model, x_train=training_data, y_train=target_data, alpha=0.01, method="greedy")

        compressed_model.compile(loss='mean_squared_error',
                                 optimizer='adam',
                                 metrics=['binary_accuracy'])

        compressed_loss[i], v = compressed_model.test_on_batch(training_data, target_data)
        logging.info("Test after compression %s" % str(compressed_loss[i]))
        logging.info("Fine tuning")
        compressed_model.train_on_batch(training_data, target_data)
        fine_tuned_loss[i], v = compressed_model.test_on_batch(training_data, target_data)
        logging.info("Test after fine tuning %s" % str(fine_tuned_loss[i]))

    plt.figure()
    plt.plot(alphas, np.repeat(initial_loss, n))
    plt.plot(alphas, compressed_loss)
    plt.plot(alphas, fine_tuned_loss)
    plt.xlabel("Alpha")
    plt.legend(["initial loss", "compressed loss", "fine tuned loss"])
    plt.show()
