import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Conv2D, MaxPooling2D,Flatten
from numpy.random import default_rng

def net(X_train, y_train, X_test, y_test, epochs, weights=None, verbose=1,dataset='MNIST',num_classes=10):
    '''Create the Neural Network'''

    if dataset == 'MNIST':

        model = Sequential()

        model.add(Dense(1000, input_shape=(784,)))
        model.add(Activation('relu'))

        model.add(Dense(1000))
        model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(Activation('softmax'))
        # compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    elif dataset == 'CIFAR-10':
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if type(weights) == np.ndarray:
        model.set_weights(weights)
        # print('Initialize with FL weights')

    # fit model
    history = model.fit(X_train, y_train,
                        batch_size=128, epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=0)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_argmax = np.argmax(y_test, axis=1)
    acc = np.mean(y_pred == y_test_argmax)

    if verbose == 1:
        print(f'Accuracy of the model: {acc:.3f}')

    return model, model.get_weights(), history, acc

def net_adversary(X_train, y_train, X_test, y_test, epochs, weights=None, verbose=1,dataset='MNIST',num_classes=10):
    '''Create the Neural Network'''

    if dataset == 'MNIST':

        model = Sequential()

        model.add(Dense(1000, input_shape=(784,)))
        model.add(Activation('relu'))

        model.add(Dense(1000))
        model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(Activation('softmax'))
        # compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    elif dataset == 'CIFAR-10':
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if type(weights) == np.ndarray:
        model.set_weights(weights)
        # print('Initialize with FL weights')

    # fit model
    history = model.fit(X_train, y_train,batch_size=128, epochs=epochs,validation_data=(X_test, y_test),verbose=0)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_argmax = np.argmax(y_test, axis=1)
    acc = np.mean(y_pred == y_test_argmax)

    if verbose == 1:
        print(f'Accuracy of the model: {acc:.3f}')

    return model, model.get_weights(), history, acc
