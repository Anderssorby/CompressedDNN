import os

import keras.backend as K
import numpy as np
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Deconv2D, UpSampling2D
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

import odin
import odin.misc.dataset.map as map_data_utils
import odin.plot as oplt
from .keras_base import KerasModelWrapper


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x


def lambda_output(input_shape):
    return input_shape[:2]


def conv_block_unet(x, f, bn_axis, bn=True, strides=(2, 2), name=""):
    x = LeakyReLU(0.2)(x)
    x = Conv2D(f, (3, 3), strides=strides, name=name, padding="same")(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)

    return x


def up_conv_block_unet(x, x2, f, bn_axis, bn=True, dropout=False, name=""):
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(f, (3, 3), name=name, padding="same")(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Concatenate(axis=bn_axis)([x, x2])

    return x


def deconv_block_unet(x, x2, f, bn_axis, bn=True, dropout=False, name=""):
    x = Activation("relu")(x)
    x = Deconv2D(f, (3, 3), strides=(2, 2), padding="same", name=name)(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Concatenate(axis=bn_axis)([x, x2])

    return x


class Generator(Model):
    def __init__(self, img_dim, architecture="upsampling"):
        unet_input = Input(shape=img_dim, name="unet_input")

        if architecture == "upsampling":
            nb_filters = 64

            if K.image_dim_ordering() == "channels_first":
                bn_axis = 1
                nb_channels = img_dim[0]
                min_s = min(img_dim[1:])
            else:
                bn_axis = -1
                nb_channels = img_dim[-1]
                min_s = min(img_dim[:-1])

            # Prepare encoder filters
            nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
            list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

            # Encoder
            list_encoder = [Conv2D(list_nb_filters[0], (3, 3),
                                   strides=(2, 2), name="unet_conv2D_1", padding="same")(unet_input)]
            for i, filters in enumerate(list_nb_filters[1:]):
                name = "unet_conv2D_%s" % (i + 2)
                conv = conv_block_unet(list_encoder[-1], filters, bn_axis, name=name)
                list_encoder.append(conv)

            # Prepare decoder filters
            list_nb_filters = list_nb_filters[:-2][::-1]
            if len(list_nb_filters) < nb_conv - 1:
                list_nb_filters.append(nb_filters)

            # Decoder
            list_decoder = [up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                                               list_nb_filters[0], bn_axis, dropout=True, name="unet_upconv2D_1")]
            for i, filters in enumerate(list_nb_filters[1:]):
                name = "unet_upconv2D_%s" % (i + 2)
                # Dropout only on first few layers
                conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], filters, bn_axis,
                                          dropout=i < 2, name=name)
                list_decoder.append(conv)

            x = Activation("relu")(list_decoder[-1])
            x = UpSampling2D(size=(2, 2))(x)
            x = Conv2D(nb_channels, (3, 3), name="last_conv", padding="same")(x)
            x = Activation("tanh")(x)
        elif architecture == "deconv":
            assert K.backend() == "tensorflow", "Not implemented with theano backend"

            nb_filters = 64
            bn_axis = -1
            nb_channels = img_dim[-1]
            min_s = min(img_dim[:-1])

            # Prepare encoder filters
            nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
            list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

            # Encoder
            list_encoder = [Conv2D(list_nb_filters[0], (3, 3),
                                   strides=(2, 2), name="unet_conv2D_1", padding="same")(unet_input)]
            # update current "image" h and w
            for i, filters in enumerate(list_nb_filters[1:]):
                name = "unet_conv2D_%s" % (i + 2)
                conv = conv_block_unet(list_encoder[-1], filters, bn_axis, name=name)
                list_encoder.append(conv)

            # Prepare decoder filters
            list_nb_filters = list_nb_filters[:-1][::-1]
            if len(list_nb_filters) < nb_conv - 1:
                list_nb_filters.append(nb_filters)

            # Decoder
            list_decoder = [deconv_block_unet(list_encoder[-1], list_encoder[-2],
                                              list_nb_filters[0], bn_axis, dropout=True, name="unet_upconv2D_1")]
            for i, filters in enumerate(list_nb_filters[1:]):
                name = "unet_upconv2D_%s" % (i + 2)
                # Dropout only on first few layers
                conv = deconv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], filters, bn_axis, dropout=i < 2,
                                         name=name)
                list_decoder.append(conv)

            x = Activation("relu")(list_decoder[-1])
            # o_shape = (batch_size,) + img_dim
            x = Deconv2D(nb_channels, (3, 3), strides=(2, 2), padding="same")(x)
            x = Activation("tanh")(x)
        else:
            raise ValueError(architecture)

        super(Generator, self).__init__(inputs=[unet_input], outputs=[x])


class DCGANDiscriminator(Model):

    def __init__(self, img_dim, nb_patch, model_name="DCGAN_discriminator", use_mbd=True):
        """
        Discriminator model of the DCGAN

        args : img_dim (tuple of int) num_chan, height, width
               pretr_weights_file (str) file holding pre trained weights

        returns : model (keras NN) the Neural Net model
        """

        list_input = [Input(shape=img_dim, name="disc_input_%s" % i) for i in range(nb_patch)]

        if K.image_dim_ordering() == "channels_first":
            bn_axis = 1
        else:
            bn_axis = -1

        nb_filters = 64
        nb_conv = int(np.floor(np.log(img_dim[1]) / np.log(2)))
        list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

        # First conv
        x_input = Input(shape=img_dim, name="discriminator_input")
        x = Conv2D(list_filters[0], (3, 3), strides=(2, 2), name="disc_conv2d_1", padding="same")(x_input)
        x = BatchNormalization(axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

        # Next convs
        for i, f in enumerate(list_filters[1:]):
            name = "disc_conv2d_%s" % (i + 2)
            x = Conv2D(f, (3, 3), strides=(2, 2), name=name, padding="same")(x)
            x = BatchNormalization(axis=bn_axis)(x)
            x = LeakyReLU(0.2)(x)

        x_flat = Flatten()(x)
        x = Dense(2, activation="softmax", name="disc_dense")(x_flat)

        patch_gan = Model(inputs=[x_input], outputs=[x, x_flat], name="PatchGAN")

        x = [patch_gan(patch)[0] for patch in list_input]
        x_mbd = [patch_gan(patch)[1] for patch in list_input]

        if len(x) > 1:
            x = Concatenate(axis=bn_axis)(x)
        else:
            x = x[0]

        if use_mbd:
            if len(x_mbd) > 1:
                x_mbd = Concatenate(axis=bn_axis)(x_mbd)
            else:
                x_mbd = x_mbd[0]

            num_kernels = 100
            dim_per_kernel = 5

            M = Dense(num_kernels * dim_per_kernel, use_bias=False, activation=None)
            MBD = Lambda(minb_disc, output_shape=lambda_output)

            x_mbd = M(x_mbd)
            x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
            x_mbd = MBD(x_mbd)
            x = Concatenate(axis=bn_axis)([x, x_mbd])

        h, w, _ = img_dim
        aux_param = Dense(2*h*w, activation="softmax", name="auxiliary_dist")(x)
        self.auxiliary = Model(inputs=list_input, outputs=aux_param, name="auxiliary_dist")

        def vmir(code, y_pred):
            """

            :param code: The code batch we are learning
            :param y_pred:
            :return:
            """
            pred_dist = stats.norm(loc=y_pred[:h*w], scale=y_pred[h*w:2*h*w])
            samples = [pred_dist.pdf(c) for c in code]
            return K.mean(samples)

        self.auxiliary.compile(Adam(), loss=vmir)

        x_out = Dense(2, activation="softmax", name="disc_output")(x)

        super(DCGANDiscriminator, self).__init__(inputs=list_input, outputs=[x_out], name=model_name)
        self.patch_gan = patch_gan


class DCGAN(Model):
    def __init__(self, generator, discriminator_model, img_dim, patch_size, image_dim_ordering):
        gen_input = Input(shape=img_dim, name="DCGAN_input")

        generated_image = generator(gen_input)

        if image_dim_ordering == "channels_first":
            h, w = img_dim[1:]
        else:
            h, w = img_dim[:-1]
        ph, pw = patch_size

        list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h // ph)]
        list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w // pw)]

        list_gen_patch = []
        for row_idx in list_row_idx:
            for col_idx in list_col_idx:
                if image_dim_ordering == "channels_last":
                    x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(generated_image)
                else:
                    x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])(generated_image)
                list_gen_patch.append(x_patch)

        output = discriminator_model(list_gen_patch)

        super(DCGAN, self).__init__(inputs=[gen_input],
                                    outputs=[generated_image, output],
                                    name="DCGAN")


class Pix2Pix(KerasModelWrapper):
    discriminator: DCGANDiscriminator
    generator: Generator
    model_name = "pix2pix_dcgan"
    dataset_name = "satellite_images"

    def __init__(self, bn_mode=False, use_mbd=False, label_flipping=0.01, label_smoothing=True,
                 dummy=False, discriminator_weights="", generator_weights="", info_gan=False,
                 **kwargs):
        """

        :param batch_size:
        :param bn_mode:
        :param use_mbd:
        :param label_flipping:
        :param label_smoothing:
        :param dummy: Use dummy data for faster debugging
        :param kwargs:
        """
        self.batch_size = kwargs.get("batch_size", 100)
        self.generator_arch = kwargs.get("generator_architecture", "deconv")
        self.image_data_format = "channels_last"
        self.img_shape = (256, 256, 3)
        self.patch_size = (64, 64)
        self.bn_mode = bn_mode
        self.use_mbd = use_mbd
        self.do_plot = kwargs.get("plot", True)
        self.n_batch_per_epoch = kwargs.get("n_batch_per_epoch", 10)
        self.label_smoothing = label_smoothing
        self.label_flipping = label_flipping
        self.epochs = kwargs.get("epochs", 100)
        self.dummy = dummy
        self.info_gan = info_gan

        self.latent_code_shape = 10

        self.discriminator_weights = discriminator_weights
        self.generator_weights = generator_weights

        super(Pix2Pix, self).__init__(**kwargs)

        self.callback_manager.add_callbacks([
            odin.callbacks.ProgbarLogger(count_mode='samples',
                                         stateful_metrics=["d_loss", "g_loss", "g_l1_loss", "g_log_loss"]),
            odin.callbacks.GANSampler(period=kwargs.get("sample_interval", 1))
        ])

        dataset_size = self.dataset.size
        steps_per_epoch = dataset_size // self.batch_size
        self.callback_manager.set_params({
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'verbose': 1,
            'samples': dataset_size,
            'steps': steps_per_epoch,
            'do_validation': False,
            'metrics': ["d_loss", "g_loss", "g_l1_loss", "g_log_loss"],
        })

        self.n_batch_per_epoch = min(steps_per_epoch, self.n_batch_per_epoch)

    def construct(self):
        if self.info_gan:
            # Add an extra code channel
            h, w, c = self.img_shape
            gen_input_shape = (h, w, c + 1)
        else:
            gen_input_shape = self.img_shape
        # Get the number of non overlapping patch and the size of input image to the discriminator
        nb_patch, img_dim_disc = map_data_utils.get_nb_patch(self.img_shape, self.patch_size, self.image_data_format)

        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
        self.generator = Generator(gen_input_shape, architecture=self.generator_arch)
        # Load discriminator model
        self.discriminator = DCGANDiscriminator(img_dim_disc, nb_patch, use_mbd=self.use_mbd)


        self.generator.compile(loss='mae', optimizer=opt_discriminator)
        self.discriminator.trainable = False

        gen_input = Input(shape=gen_input_shape, name="DCGAN_input")

        generated_image = self.generator(gen_input)

        if self.image_data_format == "channels_first":
            h, w = self.img_shape[1:]
        else:
            h, w = self.img_shape[:-1]
        ph, pw = self.patch_size

        # Split the image into patches for the discriminator
        list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h // ph)]
        list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w // pw)]

        list_gen_patch = []
        for r1, r2 in list_row_idx:
            for c1, c2 in list_col_idx:
                if self.image_data_format == "channels_last":
                    x_patch = Lambda(lambda z: z[:, r1:r2, c1:c2, :])(generated_image)
                else:
                    x_patch = Lambda(lambda z: z[:, :, r1:r2, c1:c2])(generated_image)
                list_gen_patch.append(x_patch)

        output = self.discriminator(list_gen_patch)

        model = Model(inputs=[gen_input],
                      outputs=[generated_image, output],
                      name="DCGAN")

        if self.verbose:
            print("Discriminator summary")
            self.discriminator.summary()

            print("PatchGAN summary")

            self.discriminator.patch_gan.summary()

            print("Generator summary")
            self.generator.summary()

        if self.do_plot:
            from keras.utils import plot_model
            for model in [self.discriminator, self.generator, model]:
                path = odin.check_or_create_dir(self.model_path, "figures")
                plot_model(model, to_file=path + "/%s.png" % model.name, show_shapes=True,
                           show_layer_names=True)

        loss = [l1_loss, 'binary_crossentropy']


        info_lambda = 1.0

        loss_weights = [1E1, 1]
        model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        return model

    def load(self):
        dcgan = self.construct()

        self.generator.load_weights(self.find_file(self.generator_weights))
        self.discriminator.load_weights(self.find_file(self.discriminator_weights))

        return dcgan

    def load_dataset(self):
        dataset = map_data_utils.MapImageData(limit=self.dataset_limit)
        if self.dummy:
            dataset.load_dummy(self.batch_size * 2)
        else:
            dataset.load()
        return dataset

    def save(self, epoch=None, is_checkpoint=False, **kwargs):

        if not epoch:
            epoch = self.epochs

        gen_weights_path = os.path.join(self.model_path,
                                        "gen_weights_epoch:{epoch}-loss:{g_loss:.3f}.h5".format(epoch=epoch, **kwargs))
        self.generator.save_weights(gen_weights_path, overwrite=True)

        disc_weights_path = os.path.join(self.model_path,
                                         'disc_weights_epoch:{epoch}-loss:{d_loss:.3f}.h5'.format(epoch=epoch,
                                                                                                  **kwargs))
        self.discriminator.save_weights(disc_weights_path, overwrite=True)

        dcgan_weights_path = os.path.join(self.model_path,
                                          'DCGAN_weights_epoch:{epoch}.h5'.format(epoch=epoch, **kwargs))
        self.model.save_weights(dcgan_weights_path, overwrite=True)

    def test(self, **options):
        epoch = -1
        num = 20
        _, _, x_full_val, x_sketch_val = self.dataset
        x_full_batch = x_full_val[:num]
        x_sketch_batch = x_sketch_val[:num]

        self.plot_generated_batch(x_full_batch, x_sketch_batch, "validation", epoch)

        x_disc, y_disc = self.get_disc_batch(x_full_batch,
                                             x_sketch_batch,
                                             generate=True)

        disc_loss = self.discriminator.test_on_batch(x_disc, y_disc)

        x_gen_target, x_gen = next(self.dataset.random_batch_generator(num, data_type="test"))
        y_gen = np.zeros((x_gen.shape[0], 2), dtype=np.uint8)
        y_gen[:, 1] = 1

        gen_loss = self.model.test_on_batch(x_gen, [x_gen_target, y_gen])

        return disc_loss, gen_loss

    def get_disc_batch(self, x_full_batch, x_sketch_batch, generate: bool):
        # Create x_disc: alternatively only generated or real images
        if generate:
            # Produce an output
            x_disc = self.generator.predict(x_sketch_batch)
            y_disc = np.zeros((x_disc.shape[0], 2), dtype=np.uint8)
            y_disc[:, 0] = 1

        else:
            x_disc = x_full_batch
            y_disc = np.zeros((x_disc.shape[0], 2), dtype=np.uint8)
            if self.label_smoothing:
                y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
            else:
                y_disc[:, 1] = 1

        if self.label_flipping > 0:
            p = np.random.binomial(1, self.label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

        # Now extract patches form x_disc
        x_disc = map_data_utils.extract_patches(x_disc, self.image_data_format, self.patch_size)

        return x_disc, y_disc

    def plot_generated_batch(self, x_target, y_condition, suffix,
                             epoch):
        # Generate images
        x_gen = self.generator.predict(y_condition)

        y_condition = map_data_utils.inverse_normalization(y_condition)
        x_target = map_data_utils.inverse_normalization(x_target)
        x_gen = map_data_utils.inverse_normalization(x_gen)

        rows = 6
        cols = 6
        subset = (rows * cols) // 3

        xs = y_condition[:subset]
        xg = x_gen[:subset]
        xr = x_target[:subset]

        x = np.concatenate((xs, xg, xr), axis=0)

        fig = oplt.figure(figsize=(rows, cols))

        gs1 = oplt.gridspec.GridSpec(rows, cols, figure=fig, wspace=0, hspace=0)
        for i, xs in enumerate(x, start=0):
            ax = oplt.subplot(gs1[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
            if xs.shape[-1] == 1:
                ax.imshow(xs[:, :, 0], cmap="gray")
            else:
                ax.imshow(xs)

        oplt.savefig(
            odin.check_or_create_dir(self.model_path,
                                     "figures",
                                     file="{epoch}_current_batch_{suffix}.png".format(suffix=suffix,
                                                                                      epoch=epoch)))
        oplt.show()
        oplt.clf()
        oplt.close()

    def train(self, **kwargs):
        """
        Train model

        Load the whole train data in memory for faster operations

        args: **kwargs (dict) keyword arguments that specify the model hyper parameters
        """

        # Load and rescale data
        # x_target, y_input_condition, x_target_val, y_input_condition_val = self.dataset

        self.callback_manager.on_train_begin()
        # Start training
        print("Start training")
        for epoch in range(self.epochs):
            self.callback_manager.on_epoch_begin(epoch)
            avg_gen_loss = 0
            avg_disc_loss = 0

            for batch, (x_full_batch, x_sketch_batch) in enumerate(
                    self.dataset.random_batch_generator(self.batch_size, data_type="train"), start=1):
                self.callback_manager.on_batch_begin(batch)

                # Create a batch to feed the discriminator model
                x_disc, y_disc = self.get_disc_batch(x_full_batch,
                                                     x_sketch_batch,
                                                     # alternate between generated and real samples
                                                     generate=batch % 2 == 0)

                # Update the discriminator
                disc_loss = self.discriminator.train_on_batch(x_disc, y_disc)
                avg_disc_loss += disc_loss

                # Create a batch to feed the generator model
                x_gen_target, x_gen = next(self.dataset.random_batch_generator(self.batch_size, data_type="train"))
                y_gen = np.zeros((x_gen.shape[0], 2), dtype=np.uint8)
                y_gen[:, 1] = 1

                if self.info_gan:
                    # Sample code
                    code = np.random.uniform(0, 1, self.img_shape[:-2])
                    x_gen = np.concatenate(x_gen, code, axis=2)

                # Freeze the discriminator
                self.discriminator.trainable = False
                gen_loss = self.model.train_on_batch(x_gen, [x_gen_target, y_gen])
                avg_gen_loss += gen_loss[0]
                # Train the auxiliary
                if self.info_gan:
                    gen_loss = self.discriminator.auxiliary.train_on_batch(x_gen, [x_gen_target, code])
                    # self.variational_mutual_information_regularizer(code, x_gen)

                # Unfreeze the discriminator
                self.discriminator.trainable = True

                # Save images for visualization
                # if batch % (self.n_batch_per_epoch / 2) == 0:
                #     # Get new images from validation
                #     self.plot_generated_batch(x_full_batch, x_sketch_batch, "training",
                #                               epoch)
                #     x_full_batch, x_sketch_batch = next(
                #         map_data_utils.random_batch(x_target_val, y_input_condition_val, self.batch_size))
                #     self.plot_generated_batch(x_full_batch, x_sketch_batch, "validation",
                #                               epoch)

                self.callback_manager.on_batch_end(batch, logs={"g_loss": gen_loss[0], "d_loss": disc_loss,
                                                                "g_l1_loss": gen_loss[1], "g_log_loss": gen_loss[2],
                                                                "size": self.batch_size})

                if batch >= self.n_batch_per_epoch:
                    break

            avg_disc_loss = avg_disc_loss / self.n_batch_per_epoch
            avg_gen_loss = avg_gen_loss / self.n_batch_per_epoch

            self.callback_manager.on_epoch_end(epoch + 1, logs={"g_loss": avg_gen_loss, "d_loss": avg_disc_loss})

        self.callback_manager.on_train_end()
