from keras.models import Sequential, Model
from keras.layers import Deconv2D, BatchNormalization, Conv2D, Input, Reshape, Dense, UpSampling2D
from keras.datasets import cifar10
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

    def __init__(self, latent_dim):
        """

        :param latent_dim: A suitable shape for the noise
        """
        model = Sequential(name=self.name)
        channels = 3

        base_dim = 8  # 32 // 4

        model.add(Dense(128 * base_dim * base_dim, activation="relu", input_dim=latent_dim[0]))
        model.add(Reshape((base_dim, base_dim, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same", activation="relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same", activation="relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(channels, kernel_size=4, padding="same", activation="tanh"))

        # model.add(Deconv2D(filters=256, kernel_size=4, strides=1, input_shape=latent_dim, padding="same"))
        # model.add(BatchNormalization())
        # model.add(Deconv2D(filters=128, kernel_size=4, strides=2, padding="same"))
        # model.add(BatchNormalization())
        # model.add(Deconv2D(filters=64, kernel_size=4, strides=2, padding="same"))
        # model.add(BatchNormalization())
        # model.add(Deconv2D(filters=3, kernel_size=4, strides=2, activation="tanh", padding="same"))

        noise = Input(shape=latent_dim, name="generator_input")
        img = model(noise)

        super(Generator, self).__init__(inputs=noise, outputs=img, name=self.name)


class Discriminator(Model):
    """
    (None, 32, 32, 3) -> (1)
    Tries to distinguish samples from the generator and from the real dataset
    """

    name = "WGAN-Discriminator"

    def __init__(self, img_shape):
        model = Sequential(name=self.name)
        model.add(Conv2D(filters=64, kernel_size=4, strides=2))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=4, strides=2))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=4, strides=2))
        model.add(Conv2D(filters=1, kernel_size=2, strides=1))

        img = Input(shape=img_shape, name="discriminator_input")
        validity = model(img)

        super(Discriminator, self).__init__(inputs=img, outputs=validity, name=self.name)


class WGANKerasWrapper(KerasModelWrapper):
    discriminator: Discriminator
    generator: Generator

    model_name = "cifar10_wgan"
    dataset_name = "cifar10"

    def __init__(self, latent_dim=100, clip_value=0.01, n_critic=5, initial_learning_rate=0.005, **kwargs):
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
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.learning_rate = initial_learning_rate
        self.combined = None  # alias of model

        super(WGANKerasWrapper, self).__init__(**kwargs)

    def construct(self):
        self.generator = Generator(self.latent_dim)
        self.discriminator = Discriminator(self.img_shape)

        optimizer = RMSprop(lr=self.learning_rate)

        self.discriminator.compile(loss=self.wasserstein_loss,
                                   optimizer=optimizer,
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
                              optimizer=optimizer,
                              metrics=['accuracy'])

        return self.combined

    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, epochs=1000, sample_interval=50, batch_size=1000, **options):
        # Load the dataset
        (x_train, _), (_, _) = self.load_dataset()
        self.show(format="png")
        n_critic = 1

        # Rescale -1 to 1: [0, 255] -> [-1, 1]
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        # x_train = np.expand_dims(x_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

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

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

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
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        oplt.save(category="sample_images", name="cifar10_%d" % epoch, figure=fig)
        oplt.close()

    def load_dataset(self):
        return cifar10.load_data()
