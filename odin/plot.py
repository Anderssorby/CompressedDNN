import os

import matplotlib as mat
if os.getenv("DISPLAY") is None:
    mat.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.pyplot import *

import odin
from matplotlib import rc

# Use LaTex
rc('text', usetex=True)
rc('font', family='serif')


def plot_eigen_values(eigen_values, title=""):
    fig = plt.figure()
    plt.loglog(eigen_values)
    plt.title(title + " - Eigen values (%d)" % len(eigen_values))
    plt.draw()
    return fig


def plot_matrix(matrix, title=""):
    fig = plt.figure()
    plt.matshow(matrix)
    plt.title(title)
    plt.draw()
    return fig


def plot_line(x, y, title=""):
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.draw()
    return fig


def plot_model_history(model_wrapper):
    history = model_wrapper.get_group("training_history")["history"]
    plt.figure()
    plt.plot(history['binary_accuracy'])
    # plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.figure()
    plt.plot(history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')


def labels(x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def save(name="figure"):
    path = odin.default_save_path(name, category="plots")
    plt.savefig(path + ".pdf", format="pdf")


def show(*args):
    plt.show(*args)


def draw_loss_curve(logfile, outfile, epoch=2):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for line in open(logfile):
        line = line.strip()
        if 'epoch:' not in line or 'inf' in line or 'nan' in line:
            continue
        epoch = int(re.search('epoch:([0-9]+)', line).groups()[0])
        loss = float(re.search('loss:([0-9\.]+)', line).groups()[0])
        acc = float(re.search('accuracy:([0-9\.]+)', line).groups()[0])
        if 'train' in line:
            train_loss.append([epoch, loss])
            train_acc.append([epoch, acc])
        if 'test' in line:
            test_loss.append([epoch, loss])
            test_acc.append([epoch, acc])

    train_loss = np.asarray(train_loss)
    test_loss = np.asarray(test_loss)
    train_acc = np.asarray(train_acc)
    test_acc = np.asarray(test_acc)

    if epoch < 2:
        return

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(train_loss[:, 0], train_loss[:, 1],
             label='training loss', c='g')
    # ax1.plot(test_loss[:, 0], test_loss[:, 1],
    #  label='test loss', c='g')
    ax1.set_xlim([1, len(train_loss)])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    ax2 = ax1.twinx()
    ax2.plot(train_acc[:, 0], train_acc[:, 1],
             label='training accuracy', c='r')
    ax2.plot(test_acc[:, 0], test_acc[:, 1],
             label='test accuracy', c='c')
    ax2.set_xlim([1, len(train_loss)])
    ax2.set_ylabel('accuracy')

    ax1.legend(bbox_to_anchor=(0.25, -0.1), loc=9)
    ax2.legend(bbox_to_anchor=(0.75, -0.1), loc=9)
    plt.savefig(outfile, bbox_inches='tight')
