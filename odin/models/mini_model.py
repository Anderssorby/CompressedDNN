from keras.models import Sequential
from keras.layers.core import Dense
from odin.models.keras_base import KerasModelWrapper
from odin.dataset import load_dataset


class MiniModel(KerasModelWrapper):
    model_name = 'mini_model'
    dataset_name = "mini"

    def construct(self, **kwargs):
        use_bias = kwargs.get("use_bias", False)
        model = Sequential(name="Mini")
        model.add(Dense(16, input_dim=30, activation='relu', use_bias=use_bias))
        model.add(Dense(12, activation='relu', use_bias=use_bias))
        model.add(Dense(10, activation='relu', use_bias=use_bias))
        model.add(Dense(1, activation='sigmoid', use_bias=use_bias))

        model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['binary_accuracy'])

        return model

    def load_dataset(self):
        return load_dataset("mini")
