from .base import ComputationInterface
import tensorflow as tf
from keras import backend as K
import os
import logging

accepted_layers = ['Dense', 'Conv2D', 'Conv1D']


class TensorflowWrapper(ComputationInterface):

    def calc_eigs(self, model_wrapper, exp_type, epoch_stage=15):
        model = model_wrapper.model

        path = os.path.join(self.saving_dir, model.model_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        file_start = os.path.join(path, self.args.file_prefix)
        saver = tf.train.Saver()
        for n, layer in enumerate(self.model.layers):
            if layer.__class__.__name__ not in accepted_layers:
                logging.info("Skipping...", layer)
                continue
            save_path = file_start + ".%d.ckpt" % n
            with tf.Session() as session:
                sigma_file = file_start + ".cov.%d.ckpt" % n
                if not os.path.isfile(sigma_file):
                    session = tf.Session()
                    sigma = self.compute_covariance_matrix_with_tf(model, layer, model_wrapper.x_train[:100])
                    session.run(sigma)
                    # saver.save(session, sigma_file)
                    # np.save(sigma_file, sigma)
                    # else:

                    # eig_file = file_start + ".eig.%d.ckpt" % n
                    # if not os.path.isfile(eig_file):
                    session = tf.Session()
                    eigen_values = tf.self_adjoint_eigvals(sigma)
                    session.run(sigma)
                    # saver.save(session, eig_file)

                    # eigen_values, eigen_vectors = np.linalg.eig(sigma)
                    # eigen_values = eigen_values[eigen_values != 0]
                    # eigen_values.sort()
                    # np.save(eig_file, eigen_values)

                    saver.save(session, save_path)
                else:
                    # eigen_values = np.load(eig_file)
                    saver.restore(session, save_path)
                    sigma = tf.get_variable("sigma")
                    eigen_values = tf.get_variable("eigen_values")
                sigma.eval()
                eigen_values.eval()

    @staticmethod
    def compute_covariance_matrix_with_tf(model, layer, batch):
        # intermediate_layer_model = Model(inputs=model.input,
        #                                 outputs=layer.output)
        # output = intermediate_layer_model.predict(batch)
        layer_output = K.function([model.layers[0].input],
                                  [layer.output])
        output = layer_output([batch])[0]
        flat_shape = output[0].flatten().shape[0]
        # TODO handle
        sigma = tf.zeros(shape=(flat_shape, flat_shape), dtype=tf.float32, name="sigma")
        n = len(batch)

        for i in range(n):
            g = tf.constant(output[i].flatten())
            sigma += tf.einsum('i,j->ij', g, g)
            # sigma += np.outer(g, g)
        # sigma = 1 / (n - 1) * sigma
        sigma = 1 / (n - 1) * sigma
        return sigma
