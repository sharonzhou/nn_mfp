import numpy as np
from numpy import genfromtxt

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Masking, Dropout, Activation, BatchNormalization
from keras.optimizers import Adadelta, Adam, rmsprop

import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_HOME"]="/usr/local/cuda"
os.environ["LD_LIBRARY_PATH"]="LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True 

def data():
    X = genfromtxt('first_last_filt.csv', delimiter=',')
    Y = genfromtxt('y_new.csv', delimiter=',')

    print(X.shape)
    print(Y.shape)
    print(X[0, :])

    return X, Y

def nn(X, Y):
    num_examples = X.shape[0] # all examples
    model = Sequential()
    model.add(Dense(500, input_dim=X.shape[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(400, input_dim=32))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(300, input_dim=100))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(350, input_dim=100))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(200, input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    # Train the model, iterating on the data in batches of 32 samples
    model.fit(X[:num_examples], Y[:num_examples], epochs=20, batch_size=256, validation_split=0.1)

    score, acc = model.evaluate(X[:num_examples, :], Y[:num_examples], verbose=0)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return model

X, Y = data()
model = nn(X, Y)

# serialize model to JSON
model_json = model.to_json()
with open("ff_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("ff_model.h5")
print("Saved ff model to disk")





