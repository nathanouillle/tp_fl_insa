import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pandas as pd
def load_data(dataset='MNIST'):
    '''Download the MNIST dataset and pre-process data'''
    if dataset == 'MNIST':
        return load_mnist()
    elif dataset == 'CIFAR-10':
        return load_cifar(num_classes=10)
    else :
        return None


def data_to_clients_pond(X_train, y_train, n_clients=5):
    '''Distribute data to clients'''

    if n_clients!= 5 :
        return data_to_clients_random(X_train, y_train,n_clients=n_clients)

    X_clients = []
    y_clients = []

    y_train_argmax = np.argmax(y_train, axis=1)

    mask = [(y_train_argmax == 0) | (y_train_argmax == 1),
            (y_train_argmax == 2) | (y_train_argmax == 3),
            (y_train_argmax == 4) | (y_train_argmax == 5),
            (y_train_argmax == 6) | (y_train_argmax == 7),
            (y_train_argmax == 8) | (y_train_argmax == 9)]

    for i in range(n_clients):
        X_train_client = X_train[mask[i], :]
        X_clients.append(X_train_client)

        y_train_client = y_train[mask[i], :]
        y_clients.append(y_train_client)

    return X_clients, y_clients

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

def data_to_clients_custom(X_train,y_train,n_clients=5):
    # Doesn't work
    # I get this error : ValueError: Data cardinality is ambiguous:
                        # x sizes: 1702, 400
                        # y sizes: 1702, 400
    # Don't understand why
    
    '''Distribute data to clients
    The number of clients must be 5
    You need to distribute the datas in X_train and y_train with this distribution :
        80% of the classes 0 and 1 for the first client
        .
        .
        .
        80% of the classes 8 and 9 for the last client
        20% of each remaining classes is equally split between the others clients.
        First client has 80% of classes 0 and 1, and 5% of each other classe
        X_clients and y_train are lists of n_clients lists. Each list is a client and contain the data. See data_to_clients_random for more
    No overlap
    '''
    if n_clients!= 5 :
        return data_to_clients_random(X_train, y_train,n_clients=n_clients)

    X_clients = [[] for _ in range(n_clients)]
    y_clients = [[] for _ in range(n_clients)]

    

    # Attribuer 80% des classes 0 et 1 au premier client, 80% des classes 2 et 3 au deuxième, et ainsi de suite
    for client_id in range(n_clients):
        y_train_argmax = np.argmax(y_train, axis=1)
        label_start = client_id * 2
        label_end = label_start + 1
        label_range = list(range(label_start, label_end + 1))

        # Masque pour les échantillons correspondant aux classes du client
        mask_client = np.isin(y_train_argmax, label_range)

        # Indices correspondant aux classes du client
        indices_client = np.where(mask_client)[0]

        # Mélanger les indices
        np.random.shuffle(indices_client)

        # Calculer le nombre d'échantillons à attribuer au client (80%)
        num_train_samples = int(0.8 * len(indices_client))

        # Prendre seulement 80% des échantillons
        selected_indices = indices_client[:num_train_samples]

        # Sélectionner les échantillons pour le client
        X_client = X_train[selected_indices, :]
        y_client = y_train[selected_indices, :]

        # Ajouter les échantillons au client
        X_clients[client_id].append(X_client)
        y_clients[client_id].append(y_client)

        # Supprimer les indices utilisés de X_train et y_train
        X_train = np.delete(X_train, selected_indices, axis=0)
        y_train = np.delete(y_train, selected_indices, axis=0)

        print(f'Taille X_train : {X_train.shape}')

    # Attribuer 20% des échantillons restants à chaque client
    for client_id in range(n_clients):
        # Exclure les indices d'entraînement déjà attribués
        remaining_indices = list(range(len(X_train)))

        print(f'Taille X_train : {X_train.shape}')

        # Mélanger les indices restants
        np.random.shuffle(remaining_indices)

        # Calculer le nombre d'échantillons à attribuer au client (20%)
        num_remaining_samples = int(0.2 * len(remaining_indices))

        # Prendre seulement 20% des échantillons restants
        selected_indices = remaining_indices[:num_remaining_samples]

        # Sélectionner les échantillons restants pour le client
        X_remaining = X_train[selected_indices, :]
        y_remaining = y_train[selected_indices, :]

        # Ajouter les échantillons restants au client
        X_clients[client_id].append(X_remaining)
        y_clients[client_id].append(y_remaining)

        # Supprimer les indices utilisés de X_train et y_train
        X_train = np.delete(X_train, selected_indices, axis=0)
        y_train = np.delete(y_train, selected_indices, axis=0)

    return X_clients,y_clients

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape data for the feed forward neural network
    X_train_reshaped = X_train.reshape(60000, 784)
    X_test_reshaped = X_test.reshape(10000, 784)

    # scale data
    X_train_scaled = X_train_reshaped / 255
    X_test_scaled = X_test_reshaped / 255

    # create dummy variables for the target
    y_train_dummies = np.array(pd.get_dummies(y_train))
    y_test_dummies = np.array(pd.get_dummies(y_test))

    X_train_downsized = X_train_scaled[:10000]
    X_test_downsized = X_test_scaled[:2000]

    y_train_downsized = y_train_dummies[:10000]
    y_test_downsized = y_test_dummies[:2000]

    return X_train_downsized, y_train_downsized, X_test_downsized, y_test_downsized

def load_cifar(num_classes=10):

    # reshape data for the feed forward neural network
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    # scale data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train_downsized = x_train[:4000]
    y_train_downsized = y_train[:4000]

    x_test_downsized = x_test[:1000]
    y_test_downsized = y_test[:1000]

    return x_train_downsized, y_train_downsized, x_test_downsized, y_test_downsized

