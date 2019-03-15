import keras
from keras.models import Sequential, Model
from keras.layers import Deconv2D, BatchNormalization, Conv2D, Input, Reshape, Dense, UpSampling2D, \
    Dropout, ZeroPadding2D, LeakyReLU, Flatten
from keras.datasets import mnist
from keras.optimizers import RMSprop
import keras.backend as K
import numpy as np
import odin.plot as oplt

from odin.models.keras_base import KerasModelWrapper


class Generator(Model):
    """
    noise: (latent_dim) -> (None, 32, 32, 3)
    Tries to generate samples from the distribution
    """

    name = "WGAN-Generator"

    def __init__(self, latent_dim, img_shape: (int, int, int), architecture="up_sampling"):
        """
        :param latent_dim: A suitable shape for the noise
        """
        model = Sequential(name=self.name)

        rows = img_shape[0]
        channels = img_shape[2]
        base_dim = rows // 4

        if architecture == "up_sampling":
            model.add(Dense(128 * base_dim * base_dim, activation="relu", input_dim=latent_dim[0]))
            model.add(Reshape((base_dim, base_dim, 128)))
            model.add(UpSampling2D())
            model.add(Conv2D(128, kernel_size=4, padding="same", activation="relu"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(UpSampling2D())
            model.add(Conv2D(64, kernel_size=4, padding="same", activation="relu"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Conv2D(channels, kernel_size=4, padding="same", activation="tanh"))
        elif architecture == "deconv":
            model.add(Dense(256 * base_dim * base_dim, activation="relu", input_dim=latent_dim[0]))
            model.add(Reshape((base_dim, base_dim, 256)))
            model.add(Deconv2D(filters=256, kernel_size=4, strides=1, input_shape=latent_dim, padding="same"))
            model.add(BatchNormalization())
            model.add(Deconv2D(filters=128, kernel_size=4, strides=2, padding="same"))
            model.add(BatchNormalization())
            model.add(Deconv2D(filters=64, kernel_size=4, strides=2, padding="same"))
            model.add(BatchNormalization())
            model.add(Deconv2D(filters=channels, kernel_size=4, strides=2, activation="tanh", padding="same"))
        else:
            raise ValueError(architecture)

        noise = Input(shape=latent_dim, name="generator_input")
        img = model(noise)

        super(Generator, self).__init__(inputs=noise, outputs=img, name=self.name)


class Discriminator(Model):
    """
    (None, 32, 32, 3) -> (1)
    Tries to distinguish samples from the generator and from the real dataset
    """

    name = "WGAN-Discriminator"

    def __init__(self, img_shape, architecture="up_sampling"):
        model = Sequential(name=self.name)

        if architecture == "up_sampling":
            model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
            model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(1))
        elif architecture == "deconv":
            model.add(Conv2D(filters=64, kernel_size=4, strides=2, input_shape=img_shape, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=128, kernel_size=4, strides=2, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=256, kernel_size=4, strides=2, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Conv2D(filters=1, kernel_size=2, strides=1, padding="same"))
            model.add(Flatten())
            model.add(Dense(1))
        else:
            raise ValueError(architecture)

        img = Input(shape=img_shape, name="discriminator_input")
        validity = model(img)

        super(Discriminator, self).__init__(inputs=img, outputs=validity, name=self.name)


class WGANKerasWrapper(KerasModelWrapper):
    """
    A GAN using the wasserstein metric as loss
    """

    discriminator: Discriminator
    generator: Generator

    img_rows: int
    img_cols: int
    channels: int = 3
    img_shape: (int, int, int)
    architecture: str

    def __init__(self, latent_dim=100, clip_value=0.01, n_critic=5, initial_learning_rate=0.00005,
                 architecture="up_sampling", **kwargs):
        """
        Some hyper parameters
        :param latent_dim: The dimensionality of the noise (latent_dim // 2, latent_dim // 2, 3)
        :param clip_value:
        :param initial_learning_rate:
        :param kwargs:
        """
        self.latent_dim = (latent_dim,)  # (latent_dim // 2, latent_dim // 2, 3)
        self.clip_value = clip_value
        self.n_critic = n_critic
        self.learning_rate = initial_learning_rate
        self.combined = None  # alias of model
        self.optimizer = RMSprop(lr=self.learning_rate)
        self.architecture = architecture

        super(WGANKerasWrapper, self).__init__(checkpoint_name="snapshot.{epoch:02d}-{g_loss:.2f}-{d_loss:.2f}.hdf5",
                                               **kwargs)

    def construct(self):
        self.generator = Generator(self.latent_dim, img_shape=self.img_shape, architecture=self.architecture)
        self.discriminator = Discriminator(self.img_shape, architecture=self.architecture)

        self.discriminator.compile(loss=self.wasserstein_loss,
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])
        # The generator takes noise as input and generated imgs
        z = Input(shape=self.latent_dim, name="combined_input")
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=self.optimizer,
                              metrics=['accuracy'])

        return self.combined

    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, epochs=1000, sample_interval=50, batch_size=1000, n_critic=1, **options):
        # Load the dataset
        (x_train, _), (_, _) = self.load_dataset()

        # tb_callback = keras.callbacks.TensorBoard(log_dir='./log/Graph', histogram_freq=0,
        #                                          write_graph=True, write_images=True)

        self.callback_manager.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'verbose': 1,
            'do_validation': False,
            'metrics': ["d_loss", "g_loss"],
        })

        # Rescale -1 to 1: [0, 255] -> [-1, 1]
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        # convention to add extra dimension
        if self.channels == 1:
            x_train = np.expand_dims(x_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        self.callback_manager.on_train_begin()

        for epoch in range(epochs):
            self.callback_manager.on_epoch_begin(epoch)

            noise = None
            d_loss = None

            for _ in range(n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                imgs = x_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size,) + self.latent_dim)

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            self.callback_manager.on_epoch_end(epoch, logs={"g_loss": 1 - g_loss[0], "d_loss": 1 - d_loss[0]})

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        self.callback_manager.on_train_end()

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c,) + self.latent_dim)
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = oplt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if self.channels == 1:
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                else:
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        oplt.save(category="sample_images", name=self.dataset_name + "_%d" % epoch, figure=fig)
        oplt.close()


class MnistWGAN(WGANKerasWrapper):
    model_name = "mnist_wgan"
    dataset_name = "mnist"

    img_rows = 28
    img_cols = 28
    channels = 1
    img_shape = (img_rows, img_cols, channels)

    def __init__(self, **kwargs):
        super(MnistWGAN, self).__init__(**kwargs)

    def load_dataset(self):
        return mnist.load_data()


class Cifar10WGAN(WGANKerasWrapper):
    model_name = "cifar10_wgan"
    dataset_name = "cifar10"

    img_rows = 32
    img_cols = 32
    channels = 3
    img_shape = (img_rows, img_cols, channels)

    def __init__(self, **kwargs):
        super(Cifar10WGAN, self).__init__(**kwargs)
