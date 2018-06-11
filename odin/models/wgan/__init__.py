from chainer import datasets, training, iterators, optimizers
from chainer.training import extensions

from odin.models.base import ChainerModelWrapper
from .extensions import GeneratorSample
from .iterators import RandomNoiseIterator, GaussianNoiseGenerator
from .models import Generator, Critic
from .updater import WassersteinGANUpdater


class WGANWrapper(ChainerModelWrapper):
    model_name = "cifar10_wgan"

    def __init__(self, **kwargs):
        super(WGANWrapper, self).__init__(**kwargs)
        self.generator = Generator()
        self.critic = Critic()

    def construct(self):
        pass

    def train(self, x_train=None, y_train=None, **options):
        args = options.get("args")

        # nz = args.nz
        batch_size = args.batch_size
        epochs = args.epochs
        gpu = args.gpu

        # CIFAR-10 images in range [-1, 1] (tanh generator outputs)
        train, _ = datasets.get_cifar10(withlabel=False, ndim=3, scale=2)
        train -= 1.0
        train_iter = iterators.SerialIterator(train, batch_size)

        z_iter = RandomNoiseIterator(GaussianNoiseGenerator(0, 1, args.nz),
                                     batch_size)

        optimizer_generator = optimizers.RMSprop(lr=0.00005)
        optimizer_critic = optimizers.RMSprop(lr=0.00005)
        optimizer_generator.setup(Generator())
        optimizer_critic.setup(Critic())

        updater = WassersteinGANUpdater(
            iterator=train_iter,
            noise_iterator=z_iter,
            optimizer_generator=optimizer_generator,
            optimizer_critic=optimizer_critic,
            device=gpu)

        trainer = training.Trainer(updater, stop_trigger=(epochs, 'epoch'))
        trainer.extend(extensions.ProgressBar())
        trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
        trainer.extend(GeneratorSample(), trigger=(1, 'epoch'))
        trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'critic/loss',
                                               'critic/loss/real', 'critic/loss/fake', 'generator/loss']))
        trainer.run()
