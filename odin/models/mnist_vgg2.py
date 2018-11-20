from odin.models.chainer_base import ChainerModelWrapper
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import argparse


class MLP(chainer.Chain):

    def __init__(self, n1_units, n2_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(None, n1_units),
            l2=L.Linear(None, n2_units),
            l3=L.Linear(None, n_out))
        # with self.init_scope():
        #     # the size of the inputs to each layer will be inferred
        #     self.l1 = L.Linear(None, n_units)  # n_in -> n_units
        #     self.l2 = L.Linear(None, n_units)  # n_units -> n_units
        #     self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x, multi_layer=False):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        if multi_layer:
            return h1, h2, h3
        else:
            return h3


class MNISTWrapper(ChainerModelWrapper):

    dataset_name = "mnist"
    model_name = "mnist_vgg2"

    def train(self, x_train=None, y_train=None, **options):
        args = options.get("args")
        if args.gpu >= 0:
            # Make a specified GPU current
            chainer.backends.cuda.get_device_from_id(args.gpu).use()
            self.model.to_gpu()  # Copy the model to the GPU

        # Setup an optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(self.model)

        # Load the MNIST dataset
        train, test = self.dataset

        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                     repeat=False, shuffle=False)

        # Set up a trainer
        updater = training.StandardUpdater(
            train_iter, optimizer, device=args.gpu)
        output = self.model_path
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=output)

        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(test_iter, self.model, device=args.gpu))

        # Dump a computational graph from 'loss' variable at the first iteration
        # The "main" refers to the target link of the "main" optimizer.
        trainer.extend(extensions.dump_graph('main/loss'))

        # Take a snapshot for each specified epoch
        frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
        trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

        # Write a log of evaluation statistics for each epoch
        report = extensions.LogReport()

        trainer.extend(report)

        plot = options.get("plot", getattr(args, "plot", None))
        # Save two plot images to the result dir
        if plot and extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                      'epoch', file_name='loss.png'))
            trainer.extend(
                extensions.PlotReport(
                    ['main/accuracy', 'validation/main/accuracy'],
                    'epoch', file_name='accuracy.png'))

        # Print selected entries of the log to stdout
        # Here "main" refers to the target link of the "main" optimizer again, and
        # "validation" refers to the default name of the Evaluator extension.
        # Entries other than 'epoch' are reported by the Classifier link, called by
        # either the updater or the evaluator.
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

        # Print a progress bar to stdout
        trainer.extend(extensions.ProgressBar())

        resume = options.get("resume", getattr(args, "resume", None))
        if resume:
            # Resume from a snapshot
            chainer.serializers.load_npz(args.resume, trainer)

        # Run the training
        trainer.run()

    def test(self):
        _, test = self.dataset
        return test

    def construct(self, **kwargs):
        unit = self.args.get("unit")
        l1, l2 = kwargs.get("layer_widths", (unit, unit))
        model = L.Classifier(MLP(l1, l2, 10))
        return model


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    mw = MNISTWrapper(args=args)
    mw.train(args=args)


if __name__ == "__main__":
    main()
