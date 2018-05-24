from matplotlib import pyplot as plt
import odin


def plot_eigen_values(eigen_values, title=""):
    fig = plt.figure()
    plt.loglog(eigen_values)
    plt.title(title + " - Eigen values (%d)" % len(eigen_values))
    plt.draw()
    return fig


def save(name="figure"):
    path = odin.default_save_path(name)
    plt.savefig(path)


def show(*args):
    plt.show(*args)
