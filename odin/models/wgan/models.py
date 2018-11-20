import chainer
from chainer import functions as F, datasets, optimizers, training
from chainer import links as L

from odin.models.chainer_base import ChainerModelWrapper
from odin.models.wgan.iterators import RandomNoiseIterator, GaussianNoiseGenerator
from odin.models.wgan.updater import WassersteinGANUpdater
from odin.models.wgan.extensions import GeneratorSample


class Generator(chainer.Chain):

    """(batch_size, n_z) -> (batch_size, 3, 32, 32)"""

    def __init__(self):
        super(Generator, self).__init__(
            dc1=L.Deconvolution2D(None, 256, 4, stride=1, pad=0, nobias=True),
            dc2=L.Deconvolution2D(256, 128, 4, stride=2, pad=1, nobias=True),
            dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=1, nobias=True),
            dc4=L.Deconvolution2D(64, 3, 4, stride=2, pad=1, nobias=True),
            bn_dc1=L.BatchNormalization(256),
            bn_dc2=L.BatchNormalization(128),
            bn_dc3=L.BatchNormalization(64)
        )

    def __call__(self, z, test=False):
        h = F.reshape(z, (z.shape[0], -1, 1, 1))
        chainer.using_config("test", test)
        h = F.relu(self.bn_dc1(self.dc1(h)))
        h = F.relu(self.bn_dc2(self.dc2(h)))
        h = F.relu(self.bn_dc3(self.dc3(h)))
        h = F.tanh(self.dc4(h))
        return h


class Critic(chainer.Chain):

    """(batch_size, 3, 32, 32) -> ()"""

    def __init__(self):
        super(Critic, self).__init__(
            c0=L.Convolution2D(3, 64, 4, stride=2, pad=1, nobias=True),
            c1=L.Convolution2D(64, 128, 4, stride=2, pad=1, nobias=True),
            c2=L.Convolution2D(128, 256, 4, stride=2, pad=1, nobias=True),
            c3=L.Convolution2D(256, 1, 4, stride=1, pad=0, nobias=True),
            bn_c1=L.BatchNormalization(128),
            bn_c2=L.BatchNormalization(256)
        )

    def clamp(self, lower=-0.01, upper=0.01):

        """Clamp all parameters, including the batch normalization
        parameters."""

        for params in self.params():
            params_clipped = F.clip(params, lower, upper)
            params.data = params_clipped.data

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c0(x))
        chainer.using_config("test", test)
        h = F.leaky_relu(self.bn_c1(self.c1(h)))
        h = F.leaky_relu(self.bn_c2(self.c2(h)))
        h = self.c3(h)
        h = F.sum(h) / h.size  # Mean
        return h


class WGANWrapper(ChainerModelWrapper):
    critic: Critic
    generator: Generator

    model_name = "cifar10_wgan"

    def __init__(self, **kwargs):
        super(WGANWrapper, self).__init__(**kwargs)

    def construct(self, **kwargs):
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
