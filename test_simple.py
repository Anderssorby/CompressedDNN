import matplotlib.pyplot as plt
from odin.compute.compress import CovarianceOptimizer
from odin.utils.default import default_chainer
import logging
import numpy as np
import keras

from odin.compute.lambda_param import LambdaOptimizer

if __name__ == '__main__':

    co, args, model_wrapper = default_chainer()

    logging.info("Optimizing theoretical layer width")
    lambda_optimizer = LambdaOptimizer(model_wrapper=model_wrapper)
    lambdas = lambda_optimizer.optimize()
    logging.debug("Lambdas = %s" % str(lambdas))

    n = 10
    alphas = np.linspace(0.01, 0.1, num=n)

    initial_loss, v = model_wrapper.model.test_on_batch(model_wrapper.x_test, model_wrapper.y_test)
    logging.info("Test before compression %s" % str(initial_loss))

    compressed_loss = np.zeros(n)
    fine_tuned_loss = np.zeros(n)

    for i, alpha in enumerate(alphas):
        logging.info("alpha = %s" % str(alpha))

        optimizer = CovarianceOptimizer(theoretical_widths=lambdas, lmb=0.8, plot=True)

        compressed_model = keras.models.clone_model(model_wrapper.model)
        optimizer.compress(compressed_model, x_train=model_wrapper.x_train, y_train=model_wrapper.y_train, alpha=0.01,
                           method="greedy")

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
