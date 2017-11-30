import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense

import os

save_dir = os.path.join(os.getcwd(), 'models/saved_models')
model_name = 'keras_xor_trained_model.h5'
model_path = os.path.join(save_dir, model_name)

# the four different states of the XOR gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
target_data = np.array([[0],[1],[1],[0]], "float32")

if os.path.isfile(model_path):
    model = load_model(model_path)
else:
    model = Sequential()
    model.add(Dense(16, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['binary_accuracy'])

    model.fit(training_data, target_data, nb_epoch=500, verbose=2)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

# print(model.predict(training_data).round())

