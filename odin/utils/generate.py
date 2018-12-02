import os
import odin.plot as oplt
import numpy as np


def transform(x):
    if np.linalg.norm(x) < 1:
        return np.exp(x)
    else:
        return np.log(x)


def classify(x):
    if np.mean(x) > -0.6:
        return 0
    else:
        return 1


def generate_dataset():
    n = 30
    epsilon = 0.1

    size = 200
    train_size = 100

    mean = np.zeros(n)
    sigma = np.eye(n) + epsilon * np.ones((n, n))

    rand = np.abs(np.random.multivariate_normal(mean, sigma, size=size))

    fx = np.asarray([transform(x) for x in rand[0:train_size]])
    y = np.asarray([classify(x) for x in fx])

    test_fx = np.asarray([transform(x) for x in rand[train_size:-1]])
    test_y = np.asarray([classify(x) for x in test_fx])

    return (fx, y), (test_fx, test_y)


def plot():
    train, test = np.load("data/mini.npy")
    n_train = len(train[1])
    n_test = len(test[1])
    trans = np.mean

    oplt.figure()
    oplt.subplot(2, 1, 1)
    oplt.plot(np.arange(n_train), trans(train[0], axis=1))
    oplt.plot(np.arange(n_test), trans(test[0], axis=1))
    oplt.legend(["train x", "test x"])
    oplt.labels("sample X", "$\|X\|$")

    oplt.subplot(2, 1, 2)
    oplt.plot(np.arange(n_train), train[1], linestyle="steps")
    oplt.plot(np.arange(n_test), test[1], linestyle="steps")
    oplt.legend(["train y", "test y"])
    oplt.labels("sample X", "$Y$")
    oplt.tight_layout()

    oplt.save("mini_dataset")
    oplt.show()


def generate_plot_and_save():
    train, test = generate_dataset()

    if not os.path.isdir("data"):
        os.mkdir("data")

    plot()

    np.save("data/mini.npy", (train, test))
