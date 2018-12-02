from odin.models.keras_base import KerasModelWrapper
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.models import Sequential


class MNISTCNNWrapper(KerasModelWrapper):

    def construct(self):
        input_shape = (None,)
        num_category = 10
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # flatten since too many dimensions, we only want a classification output
        model.add(Flatten())
        # fully connected to get all relevant data
        model.add(Dense(128, activation='relu'))
        # one more dropout for convergence' sake :)
        model.add(Dropout(0.5))
        # output a softmax to squash the matrix into output probabilities
        model.add(Dense(num_category, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        return model
