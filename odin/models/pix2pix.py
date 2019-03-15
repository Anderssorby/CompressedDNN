import keras.backend as K
import numpy as np
import time
import keras
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Deconv2D, UpSampling2D
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import progressbar

from .keras_base import KerasModelWrapper
import odin
import odin.misc.dataset.map as data_utils


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x


def lambda_output(input_shape):
    return input_shape[:2]


# def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, dropout=False, strides=(2,2)):

#     x = Conv2D(f, (3, 3), strides=strides, name=name, padding="same")(x)
#     if bn:
#         x = BatchNormalization(axis=bn_axis)(x)
#     x = LeakyReLU(0.2)(x)
#     if dropout:
#         x = Dropout(0.5)(x)

#     return x


# def up_conv_block_unet(x1, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False):

#     x1 = UpSampling2D(size=(2, 2))(x1)
#     x = merge([x1, x2], mode="concat", concat_axis=bn_axis)

#     x = Conv2D(f, (3, 3), name=name, padding="same")(x)
#     if bn:
#         x = BatchNormalization(axis=bn_axis)(x)
#     x = Activation("relu")(x)
#     if dropout:
#         x = Dropout(0.5)(x)

#     return x

def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, strides=(2, 2)):
    x = LeakyReLU(0.2)(x)
    x = Conv2D(f, (3, 3), strides=strides, name=name, padding="same")(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)

    return x


def up_conv_block_unet(x, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False):
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(f, (3, 3), name=name, padding="same")(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Concatenate(axis=bn_axis)([x, x2])

    return x


def deconv_block_unet(x, x2, f, h, w, batch_size, name, bn_mode, bn_axis, bn=True, dropout=False):
    o_shape = (batch_size, h * 2, w * 2, f)
    x = Activation("relu")(x)
    x = Deconv2D(f, (3, 3), output_shape=o_shape, strides=(2, 2), padding="same")(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Concatenate(axis=bn_axis)([x, x2])

    return x


class Generator(Model):
    def __init__(self, img_dim, bn_mode, batch_size, architecture="upsampling"):
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
            for i, f in enumerate(list_nb_filters[1:]):
                name = "unet_conv2D_%s" % (i + 2)
                conv = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
                list_encoder.append(conv)

            # Prepare decoder filters
            list_nb_filters = list_nb_filters[:-2][::-1]
            if len(list_nb_filters) < nb_conv - 1:
                list_nb_filters.append(nb_filters)

            # Decoder
            list_decoder = [up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                                               list_nb_filters[0], "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
            for i, f in enumerate(list_nb_filters[1:]):
                name = "unet_upconv2D_%s" % (i + 2)
                # Dropout only on first few layers
                if i < 2:
                    d = True
                else:
                    d = False
                conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, name, bn_mode, bn_axis,
                                          dropout=d)
                list_decoder.append(conv)

            x = Activation("relu")(list_decoder[-1])
            x = UpSampling2D(size=(2, 2))(x)
            x = Conv2D(nb_channels, (3, 3), name="last_conv", padding="same")(x)
            x = Activation("tanh")(x)
        elif architecture == "deconv":
            assert K.backend() == "tensorflow", "Not implemented with theano backend"

            nb_filters = 64
            bn_axis = -1
            h, w, nb_channels = img_dim
            min_s = min(img_dim[:-1])

            # Prepare encoder filters
            nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
            list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

            # Encoder
            list_encoder = [Conv2D(list_nb_filters[0], (3, 3),
                                   strides=(2, 2), name="unet_conv2D_1", padding="same")(unet_input)]
            # update current "image" h and w
            h, w = h / 2, w / 2
            for i, f in enumerate(list_nb_filters[1:]):
                name = "unet_conv2D_%s" % (i + 2)
                conv = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
                list_encoder.append(conv)
                h, w = h / 2, w / 2

            # Prepare decoder filters
            list_nb_filters = list_nb_filters[:-1][::-1]
            if len(list_nb_filters) < nb_conv - 1:
                list_nb_filters.append(nb_filters)

            # Decoder
            list_decoder = [deconv_block_unet(list_encoder[-1], list_encoder[-2],
                                              list_nb_filters[0], h, w, batch_size,
                                              "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
            h, w = h * 2, w * 2
            for i, f in enumerate(list_nb_filters[1:]):
                name = "unet_upconv2D_%s" % (i + 2)
                # Dropout only on first few layers
                if i < 2:
                    d = True
                else:
                    d = False
                conv = deconv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, h,
                                         w, batch_size, name, bn_mode, bn_axis, dropout=d)
                list_decoder.append(conv)
                h, w = h * 2, w * 2

            x = Activation("relu")(list_decoder[-1])
            o_shape = (batch_size,) + img_dim
            x = Deconv2D(nb_channels, (3, 3), output_shape=o_shape, strides=(2, 2), padding="same")(x)
            x = Activation("tanh")(x)
        else:
            raise ValueError(architecture)

        super(Generator, self).__init__(inputs=[unet_input], outputs=[x])


class DCGANDiscriminator(Model):

    def __init__(self, img_dim, nb_patch, bn_mode, model_name="DCGAN_discriminator", use_mbd=True):
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

        PatchGAN = Model(inputs=[x_input], outputs=[x, x_flat], name="PatchGAN")
        print("PatchGAN summary")
        PatchGAN.summary()

        x = [PatchGAN(patch)[0] for patch in list_input]
        x_mbd = [PatchGAN(patch)[1] for patch in list_input]

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

        x_out = Dense(2, activation="softmax", name="disc_output")(x)

        super(DCGANDiscriminator, self).__init__(inputs=list_input, outputs=[x_out], name=model_name)


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

    def __init__(self, batch_size=100, bn_mode=False, use_mbd=False, label_flipping=0.01, label_smoothing=True,
                 **kwargs):
        self.batch_size = batch_size
        self.generator_arch = kwargs["generator_architecture"]
        self.image_data_format = "channels_last"
        self.img_shape = (256, 256, 3)
        self.patch_size = (64, 64)  # kwargs["patch_size"]
        self.bn_mode = bn_mode
        self.use_mbd = use_mbd
        self.do_plot = kwargs["plot"]
        self.n_batch_per_epoch = kwargs.get("n_batch_per_epoch", 10)
        self.label_smoothing = label_smoothing
        self.label_flipping = label_flipping
        self.epochs = kwargs["epochs"]

        super(Pix2Pix, self).__init__(**kwargs)

    def construct(self):

        # Get the number of non overlapping patch and the size of input image to the discriminator
        nb_patch, img_dim_disc = data_utils.get_nb_patch(self.img_shape, self.patch_size, self.image_data_format)

        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
        self.generator = Generator(self.img_shape, self.bn_mode, self.batch_size, architecture=self.generator_arch)
        # Load discriminator model
        self.discriminator = DCGANDiscriminator(img_dim_disc, nb_patch, self.bn_mode, use_mbd=self.use_mbd)

        self.generator.compile(loss='mae', optimizer=opt_discriminator)
        self.discriminator.trainable = False

        model = DCGAN(self.generator,
                      self.discriminator,
                      self.img_shape,
                      self.patch_size,
                      self.image_data_format)

        if self.do_plot:
            from keras.utils import plot_model
            for model in [self.discriminator, self.generator, model]:
                path = odin.check_or_create_dir(self.model_path, "figures")
                plot_model(model, to_file=path + "/%s.png" % model.name, show_shapes=True,
                           show_layer_names=True)

        loss = [l1_loss, 'binary_crossentropy']
        loss_weights = [1E1, 1]
        model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        return model

    def load_dataset(self):
        return data_utils.MapImageData()

    def train(self, **kwargs):
        """
        Train model

        Load the whole train data in memory for faster operations

        args: **kwargs (dict) keyword arguments that specify the model hyperparameters
        """

        # Roll out the parameters
        # model_name = kwargs["model_name"]
        # generator = kwargs["generator"]
        # img_dim = kwargs["img_dim"]
        # bn_mode = kwargs["bn_mode"]
        # label_smoothing = kwargs["use_label_smoothing"]
        # label_flipping = kwargs["label_flipping"]
        # epochs = kwargs["epochs"]

        self.callback_manager.add_callbacks([
            keras.callbacks.ProgbarLogger(count_mode='samples',
                                          stateful_metrics=["d_loss", "g_loss", "g_l1_loss", "g_log_loss"])
        ])

        # Load and rescale data
        X_full_train, X_sketch_train, X_full_val, X_sketch_val = self.dataset

        dataset_size = X_full_train.shape[0]
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
        # img_dim = X_full_train.shape[-3:]

        self.callback_manager.on_train_begin()
        # Start training
        print("Start training")
        for epoch in range(self.epochs):
            self.callback_manager.on_epoch_begin(epoch)
            # Initialize progbar and batch counter
            batch_counter = 1
            start = time.time()
            avg_gen_loss = 0
            avg_disc_loss = 0
            # progbar = progressbar.ProgressBar(maxval=steps_per_epoch)

            for batch, (X_full_batch, X_sketch_batch) in enumerate(
                    data_utils.gen_batch(X_full_train, X_sketch_train, self.batch_size)):
                self.callback_manager.on_batch_begin(batch)
                # Create a batch to feed the discriminator model
                X_disc, y_disc = data_utils.get_disc_batch(X_full_batch,
                                                           X_sketch_batch,
                                                           self.generator,
                                                           batch_counter,
                                                           self.patch_size,
                                                           self.image_data_format,
                                                           label_smoothing=self.label_smoothing,
                                                           label_flipping=self.label_flipping)

                # Update the discriminator
                disc_loss = self.discriminator.train_on_batch(X_disc, y_disc)
                avg_disc_loss += disc_loss

                # Create a batch to feed the generator model
                X_gen_target, X_gen = next(data_utils.gen_batch(X_full_train, X_sketch_train, self.batch_size))
                y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                y_gen[:, 1] = 1

                # Freeze the discriminator
                self.discriminator.trainable = False
                gen_loss = self.model.train_on_batch(X_gen, [X_gen_target, y_gen])
                avg_gen_loss += gen_loss[0]
                # Unfreeze the discriminator
                self.discriminator.trainable = True

                batch_counter += 1
                # progbar.update(batch)
                # add(self.batch_size, values=[("D logloss", disc_loss),
                #                              ("G tot", gen_loss[0]),
                #                              ("G L1", gen_loss[1]),
                #                              ("G logloss", gen_loss[2])])

                # Save images for visualization
                if batch_counter % (self.n_batch_per_epoch / 2) == 0:
                    # Get new images from validation
                    data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, self.generator,
                                                    self.batch_size, self.image_data_format, "training",
                                                    self.model_path)
                    X_full_batch, X_sketch_batch = next(data_utils.gen_batch(X_full_val, X_sketch_val, self.batch_size))
                    data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, self.generator,
                                                    self.batch_size, self.image_data_format, "validation",
                                                    self.model_path)

                self.callback_manager.on_batch_end(batch, logs={"g_loss": gen_loss[0], "d_loss": disc_loss,
                                                                "g_l1_loss": gen_loss[1], "g_log_loss": gen_loss[2]})

                if batch_counter >= self.n_batch_per_epoch:
                    break

            avg_disc_loss = avg_disc_loss / batch_counter
            avg_gen_loss = avg_gen_loss / batch_counter

            self.callback_manager.on_epoch_end(epoch, logs={"g_loss": avg_gen_loss, "d_loss": avg_disc_loss})

            print("")
            print('Epoch %s/%s, Time: %s' % (epoch + 1, self.epochs, time.time() - start))

            # if e % save_every_epoch == 0:
            #     gen_weights_path = os.path.join(self.model_path,
            #                                     'models/%s/gen_weights_epoch%s.h5' % (model_name, e))
            #     self.save_weights(gen_weights_path, overwrite=True)
            #
            #     disc_weights_path = os.path.join(self.model_path,
            #                                      'models/%s/disc_weights_epoch%s.h5' % (model_name, e))
            #     self.discriminator.save_weights(disc_weights_path, overwrite=True)
            #
            #     DCGAN_weights_path = os.path.join(self.model_path,
            #                                       'models/%s/DCGAN_weights_epoch%s.h5' % (model_name, e))
            #     DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

        self.callback_manager.on_train_end()
