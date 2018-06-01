import numpy as np
from numpy import genfromtxt

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Masking, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop

import tensorflow as tf

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_HOME"]="/usr/local/cuda"
os.environ["LD_LIBRARY_PATH"]="LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True 

def data():
    X = genfromtxt('first_last_filt_age_gender_bmi.csv', delimiter=',')
    Y = genfromtxt('y_new_age_gender_bmi.csv', delimiter=',')

    print(X.shape)
    print(Y.shape)
    print(X[0, :])

    return X, Y

def nn(X, Y):
    num_examples = X.shape[0] # all examples
    model = Sequential()
    model.add(Dense({{choice([256, 512, 1024])}}, input_dim=X.shape[1]))
    model.add(Activation({{choice(['relu', 'tanh', 'softmax'])}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'tanh', 'softmax'])}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'tanh', 'softmax'])}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'tanh', 'softmax'])}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'tanh', 'softmax'])}}))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'adadelta'])}},
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(X[:num_examples], Y[:num_examples], epochs=65, batch_size=1024, validation_split=0.1)

    score, acc = model.evaluate(X[:num_examples, :], Y[:num_examples], verbose=0)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

best_params, best_model = optim.minimize(model=nn,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=3,
                                        trials=Trials())


print("Best performing model chosen hyper-parameters:")
print(best_params)
X, Y = data()
print("Evaluation of best performing model:")
print(best_model.evaluate(X, Y))

# serialize model to JSON
model_json = best_model.to_json()
with open("ff_model_hyperparam.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
best_model.save_weights("ff_model_hyperparam.h5")
print("Saved ff model to disk")




