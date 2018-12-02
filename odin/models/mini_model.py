from keras.models import Sequential
from keras.layers.core import Dense
from odin.models.keras_base import KerasModelWrapper
from odin.dataset import load_dataset


class MiniModel(KerasModelWrapper):
    model_name = 'mini_model'
    dataset_name = "mini"

    def construct(self):
        use_bias = self.args.get("use_bias", False)

        layer_widths = self.args.get("layer_widths")
        if layer_widths is None or len(layer_widths) != 3:
            layer_widths = [16, 12, 10]

        model = Sequential(name="Mini")
        model.add(Dense(layer_widths[0], input_dim=30, activation='relu', use_bias=use_bias, name="hidden1"))
        model.add(Dense(layer_widths[1], activation='relu', use_bias=use_bias, name="hidden2"))
        model.add(Dense(layer_widths[2], activation='relu', use_bias=use_bias, name="hidden3"))
        model.add(Dense(1, activation='sigmoid', use_bias=use_bias, name="output"))

        weights = self.args.get("weights")
        if weights is not None and len(weights) == 4:
            for l, w in enumerate(weights):
                layer = model.get_layer(index=l)
                layer.set_weights([w])

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['binary_accuracy'])

        return model

    def load_dataset(self):
        return load_dataset("mini")
