from compress import WidthOptimizer
import keras
import argparse
from models import load_model
import numpy as np
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False,
                default="keras_xor", help="model to compress")
args = vars(ap.parse_args())



model, training_data, target_data = load_model(args["model"])

n = 10
alphas = np.linspace(0.01, 0.1, num=n)

initial_loss, v = model.test_on_batch(training_data, target_data)
print("Test before compression ", initial_loss)

compressed_loss = np.zeros(n)
fine_tuned_loss = np.zeros(n)

for i, alpha in enumerate(alphas):
    print("alpha =", alpha)

    optimizer = WidthOptimizer(lmb=0.8, plot=False)

    compressed_model = keras.models.clone_model(model)
    optimizer.compress(compressed_model, x_train=training_data, y_train=target_data, alpha=0.01, method="greedy")

    compressed_model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['binary_accuracy'])

    compressed_loss[i], v = compressed_model.test_on_batch(training_data, target_data)
    print("Test after compression ", compressed_loss[i])
    print("Fine tuning")
    compressed_model.train_on_batch(training_data, target_data)
    fine_tuned_loss[i], v = compressed_model.test_on_batch(training_data, target_data)
    print("Test after fine tuning ", fine_tuned_loss[i])

plt.figure()
plt.plot(alphas, np.repeat(initial_loss, n))
plt.plot(alphas, compressed_loss)
plt.plot(alphas, fine_tuned_loss)
plt.legend(["initial loss", "compressed loss", "fine tuned loss"])
plt.show()

