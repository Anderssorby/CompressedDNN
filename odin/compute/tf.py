from .base import ComputationInterface
from progress.bar import ChargingBar
import numpy as np
import tensorflow as tf
import os


class TensorflowWrapper(ComputationInterface):

    def load_elements(self, group_name, model_wrapper):
        save_path = os.path.join(model_wrapper.model_path, group_name, "saved.ckpt")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        with tf.Session() as session:

            saver = tf.train.Saver()

            saver.restore(session, save_path)

    def store_elements(self, elements, group_name, model_wrapper, use_saver=False):
        if not use_saver:
            super(TensorflowWrapper, self).store_elements(elements=elements, group_name=group_name,
                                                          model_wrapper=model_wrapper)

        save_path = os.path.join(model_wrapper.model_path, group_name, "saved.ckpt")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        with tf.Session() as session:
            for name, var in elements.items():
                if not isinstance(var, tf.Variable):
                    elements[name] = tf.Variable(var)
                else:
                    var.initializer.run()
                    var.op.run()
            saver = tf.train.Saver(elements)

            saver.save(session, save_path)
        return elements

    def calc_inter_layer_covariance(self, model_wrapper, use_training_data=True, batch_size=-1, **options):
        is_chainer = model_wrapper.model_type == "chainer"

        train, test = model_wrapper.dataset

        data_x = train if use_training_data else test

        if is_chainer:
            data_x = np.moveaxis(data_x, -1, 0)[0]
        else:
            data_x = data_x[0]

        data_x = np.stack(data_x, axis=0)
        data_size = n = len(data_x)

        if batch_size > 0:
            perm = np.random.permutation(data_size)
            data_x = data_x[perm[0:batch_size]]
            n = batch_size

        n_layers = len(model_wrapper.layers())
        bar = ChargingBar("Calculating inter layer covariance", max=n_layers)

        layer_outputs = model_wrapper.get_layer_outputs(data_x)
        to_save = {}

        for l, layer_output in enumerate(layer_outputs):
            if is_chainer:
                layer_output = layer_output.data
            flat_shape = layer_output[0].flatten().shape[0]

            sigma = tf.zeros(shape=(flat_shape, flat_shape), dtype=tf.float32, name="sigma%d" % l)

            for output in layer_output:
                g = tf.constant(output.flatten())
                sigma += tf.einsum('i,j->ij', g, g)

            sigma = tf.Variable(1 / (n - 1) * sigma, name="sigma%d" % l)

            eigen_values = tf.self_adjoint_eigvals(sigma, name="eigen_values%d" % l)
            to_save["sigma%d" % l] = sigma
            to_save["eigen_values%d" % l] = tf.Variable(eigen_values)

        bar.next()

        self.store_elements(group_name="inter_layer_covariance",
                            elements=to_save, model_wrapper=model_wrapper)
