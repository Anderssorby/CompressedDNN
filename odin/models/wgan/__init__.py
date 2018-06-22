import chainer
from chainer import datasets, training, optimizers

from odin.models.base import ChainerModelWrapper
from .extensions import GeneratorSample
from .iterators import RandomNoiseIterator, GaussianNoiseGenerator
from .models import Generator, Critic
from .updater import WassersteinGANUpdater


class WGANWrapper(ChainerModelWrapper):
    model_name = "cifar10_wgan"

    def __init__(self, **kwargs):
        super(WGANWrapper, self).__init__(**kwargs)
        self.generator = None
        self.critic = None

    def construct(self):
        self.generator = Generator()
        self.critic = Critic()

    def train(self, x_train=None, y_train=None, **options):
        kwargs = vars(options.get("args"))

        nz = kwargs.get('nz', 100)
        batch_size = kwargs.get('batch_size', 64)
        epochs = kwargs.get('epochs', 10000)
        gpu = kwargs.get('gpu', -1)

        # CIFAR-10 images in range [-1, 1] (tanh generator outputs)
        train, _ = datasets.get_cifar10(withlabel=False, ndim=3, scale=2)
        train -= 1.0
        train_iter = chainer.iterators.SerialIterator(train, batch_size)

        z_iter = RandomNoiseIterator(GaussianNoiseGenerator(0, 1, nz),
                                     batch_size)

        optimizer_generator = optimizers.RMSprop(lr=0.00005)
        optimizer_critic = optimizers.RMSprop(lr=0.00005)
        optimizer_generator.setup(self.generator)
        optimizer_critic.setup(self.critic)

        updater = WassersteinGANUpdater(
            iterator=train_iter,
            noise_iterator=z_iter,
            optimizer_generator=optimizer_generator,
            optimizer_critic=optimizer_critic,
            device=gpu)

        trainer = training.Trainer(updater, stop_trigger=(epochs, 'epoch'), out=self.model_path)
        trainer.extend(training.extensions.ProgressBar())
        trainer.extend(training.extensions.LogReport(trigger=(1, 'iteration')))
        trainer.extend(GeneratorSample(), trigger=(1, 'epoch'))
        trainer.extend(training.extensions.PrintReport(['epoch', 'iteration', 'critic/loss',
                                                        'critic/loss/real', 'critic/loss/fake', 'generator/loss']))
        trainer.run()
