import numpy as np
from numpy import genfromtxt

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Masking, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ["CUDA_HOME"]="/usr/local/cuda"
os.environ["LD_LIBRARY_PATH"]="LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/"

def data():

    # padded_X = genfromtxt('/remote/mgord/under_armour/derived_data/summary_tables/cs230_padded_05292018.csv', delimiter=',')
    # Y = genfromtxt('/remote/mgord/under_armour/derived_data/summary_tables/cs230_y_05292018.csv', delimiter=',')

    padded_X = genfromtxt('cs230_padded_05292018.csv', delimiter=',')
    Y = genfromtxt('cs230_y_05292018.csv', delimiter=',')

    print(padded_X.shape)
    print(Y.shape)

    X = padded_X.reshape(padded_X.shape + (1,))

    print(X[0, :, :])
    print(X.shape)

    return X, Y

def nn(X, Y):
    model = Sequential()
    M = Masking(mask_value=0, input_shape=(54, 1))
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


    # Debugging: Predict training examples themselves (since no validation split)...
    num_examples = 1
    model.fit(X[:num_examples, :, :], Y[:num_examples], epochs=1, batch_size=32)

    # for i in range(NUM_EXAMPLES):
    #     print(model.predict_classes(X[i].reshape(1, 54, 1)))

    score, acc = model.evaluate(X[:num_examples, :, :], Y[:num_examples], verbose=0)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

best_params, best_model = optim.minimize(model=nn,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=3,
                                        trials=Trials())

# serialize model to JSON
model_json = best_model.to_json()
with open("best_model_test.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
best_model.save_weights("best_model_test.h5")
print("Saved best test model to disk")

X, Y = data()
print("Evaluation of best performing model:")
print(best_model.evaluate(X[:1, :, :], Y[:1]))
print("Best performing model chosen hyper-parameters:")
print(best_params)




