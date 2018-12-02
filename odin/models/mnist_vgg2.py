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
        gpu = options.get("gpu")
        epoch = options.get("epoch")
        batch_size = options.get("batch_size", 128)
        frequency = options.get("frequency")

        if gpu >= 0:
            # Make a specified GPU current
            chainer.backends.cuda.get_device_from_id(gpu).use()
            self.model.to_gpu()  # Copy the model to the GPU

        # Setup an optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(self.model)

        # Load the MNIST dataset
        train, test = self.dataset

        train_iter = chainer.iterators.SerialIterator(train, batch_size)
        test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                     repeat=False, shuffle=False)

        # Set up a trainer
        updater = training.StandardUpdater(
            train_iter, optimizer, device=gpu)
        output = self.model_path
        trainer = training.Trainer(updater, (epoch, 'epoch'), out=output)

        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(test_iter, self.model, device=gpu))

        # Dump a computational graph from 'loss' variable at the first iteration
        # The "main" refers to the target link of the "main" optimizer.
        trainer.extend(extensions.dump_graph('main/loss'))

        # Take a snapshot for each specified epoch
        frequency = epoch if frequency == -1 else max(1, frequency)
        trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

        # Write a log of evaluation statistics for each epoch
        report = extensions.LogReport()

        trainer.extend(report)

        plot = options.get("plot")
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

        resume = options.get("resume", None)
        if resume:
            chainer.serializers.load_npz(resume, trainer)

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
