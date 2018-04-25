import os
from abc import abstractmethod
from keras.models import load_model
import keras.backend as K
import chainer.serializers


class ModelWrapper(object):
    save_dir = os.path.join(os.getcwd(), 'models/saved_models')

    model_name = "name_not_specified"
    model_type = "keras"

    def __init__(self):
        if os.path.isfile(self.model_path):
            self.model = load_model(self.model_path)
        else:
            self.model = self.construct()
        self.model.model_name = self.model_name

    @property
    def model_path(self):
        return os.path.join(self.save_dir, self.model_name + '_trained_model.h5')

    @abstractmethod
    def construct(self):
        pass

    def train(self, x_train=None, y_train=None, **options):
        x_train = x_train if x_train is not None else self.x_train
        y_train = y_train if y_train is not None else self.y_train

        self.model.fit(x_train, y_train, **options)

    def save(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.model.save(self.model_path)

    @property
    def dataset(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_nth_layer_output(self, n, batch):
        if self.model_type == "keras":
            # intermediate_layer_model = Model(inputs=model.input,
            #                                 outputs=layer.output)
            # output = intermediate_layer_model.predict(batch)
            layer_output = K.function([self.model.layers[0].input],
                                      [self.model.layers[n].output])
            output = layer_output([batch])[0]

            return output


class ChainerModelWrapper(ModelWrapper):
    model_type = "chainer"

    @abstractmethod
    def construct(self):
        pass

    def load(self):
        chainer.serializers.load_hdf5(self.model_path,
                                      self.construct())
