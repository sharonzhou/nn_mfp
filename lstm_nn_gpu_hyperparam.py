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
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ["CUDA_HOME"]="/usr/local/cuda"
os.environ["LD_LIBRARY_PATH"]="LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True 

def data():
    padded_X = genfromtxt('cs230_padded_05292018.csv', delimiter=',')
    Y = genfromtxt('cs230_y_05292018.csv', delimiter=',')
    X = padded_X.reshape(padded_X.shape + (1,))
    print(X.shape, Y.shape)
    return X, Y

def nn(X, Y):
    model = Sequential()
    M = Masking(mask_value=0, input_shape=(X.shape[1], X.shape[2]))
    model.add(M)
    model.add(LSTM(500, input_shape=(None, 1), activation="tanh"))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'tanh', 'softmax'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'tanh', 'softmax'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'adadelta'])}},
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


	# Train the model, iterating on the data in batches of 128 samples
    model.fit(X, Y, epochs=30, batch_size=128, validation_split=0.1)

    score, acc = model.evaluate(X, Y, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

best_params, best_model = optim.minimize(model=nn,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=100,
                                        trials=Trials())


# serialize model to JSON
model_json = best_model.to_json()
with open("best_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
best_model.save_weights("best_model.h5")
print("Saved best model to disk")

X, Y = data()
print("Evaluation of best performing model:")
print(best_model.evaluate(X, Y))
print("Best performing model chosen hyper-parameters:")
print(best_params)



