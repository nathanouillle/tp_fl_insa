import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
def load_data(num_classes = 10):
    '''Download the CIFAR-10 dataset and pre-process data'''


    # reshape data for the feed forward neural network
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    # scale data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train_downsized = x_train[:40000]
    y_train_downsized = y_train[:40000]

    x_test_downsized = x_test[:5000]
    y_test_downsized = y_test[:5000]

    x_test_car = []
    y_test_car = []

    for X, y in zip(x_test_downsized, y_test_downsized):
        if np.argmax(y) == 1:
            y_test_car.append(y)
            x_test_car.append(X)

    x_test_car, y_test_car = np.array(x_test_car), np.array(y_test_car)

    return x_train_downsized, y_train_downsized, x_test_downsized, y_test_downsized, x_test_car,y_test_car



def data_to_clients_random(X_train, y_train,n_clients=4):
    '''Distribute data to clients'''
    X_clients = []
    y_clients = []

    step = len(X_train)//n_clients
    for i in range(n_clients):
        X_train_client = X_train[i*step:(i+1)*step]
        X_clients.append(X_train_client)

        y_train_client = y_train[i*step:(i+1)*step]
        y_clients.append(y_train_client)

    return X_clients, y_clients