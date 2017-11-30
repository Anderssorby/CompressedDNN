from compress import WidthOptimizer

import argparse
from models import load_model

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False,
                default="keras_xor", help="model to compress")
args = vars(ap.parse_args())



model, training_data, target_data = load_model(args["model"])

print("Test before compression ", model.test_on_batch(training_data, target_data))

optimizer = WidthOptimizer(model, x_train=training_data, y_train=target_data, alpha=0.1, lmb=0.8)

optimizer.compress(method="greedy")

model.train_on_batch(training_data, target_data)

print("Test after compression ", model.test_on_batch(training_data, target_data))
