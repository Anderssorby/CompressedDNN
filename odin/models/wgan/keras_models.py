from keras.models import Sequential
from keras.layers import Dense, Deconv2D, BatchNormalization, Conv2D
import numpy as np
import odin.plot as oplt

from odin.models.keras_base import KerasModelWrapper


class Generator(Sequential):

    def __init__(self):
        super(Generator, self).__init__().__init__(name="WGAN-Generator")
        self.add(Deconv2D(None, 256, filters=4, strides=1))
        self.add(BatchNormalization(256))
        self.add(Deconv2D(256, 128, filters=4, strides=2))
        self.add(BatchNormalization(128))
        self.add(Deconv2D(128, 64, filters=4, strides=2))
        self.add(BatchNormalization(64))
        self.add(Deconv2D(64, 3, filters=4, strides=2, activation="tanh"))


class Discriminator(Sequential):

    def __init__(self):
        super(Discriminator, self).__init__().__init__(name="WGAN-Discriminator")
        self.add(Conv2D(3, 64, filters=4, strides=2))
        self.add(BatchNormalization(64))
        self.add(Conv2D(64, 128, filters=4, strides=2))
        self.add(BatchNormalization(128))
        self.add(Conv2D(128, 256, filters=4, strides=2))
        self.add(Conv2D(256, 1, filters=4, strides=1))


class WGANKerasWrapper(KerasModelWrapper):
    discriminator: Discriminator
    generator: Generator

    model_name = "cifar10_wgan"
    dataset_name = "cifar10"

    def __init__(self, **kwargs):
        super(WGANKerasWrapper, self).__init__(**kwargs)

    def construct(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.model = self.discriminator

    def train(self, epochs=1000, sample_interval=50, batch_size=1000, **options):
        # Load the dataset
        (x_train, _), (_, _) = self.load_dataset()

        n_critic = 1

        # Rescale -1 to 1
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = np.expand_dims(x_train, axis=3)

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
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

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
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = oplt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        oplt.save("images/cifar10_%d.png" % epoch)
        oplt.close()
