import numpy as np
from aggregators import aggregator_mean,aggregator_median,aggregator_custom
from train import net,net_adversary

def federated(X_clients, y_clients, X_test, y_test, fl_iterations,n_adversaire=0,epochs_FL=1,dataset='MNIST'):
    '''This function performs the Federated Learning'''
    model_init = net(X_clients[0], y_clients[0], X_test, y_test, epochs=1, weights=False, verbose=0,dataset=dataset)
    acc_fl = []

    adversaire = [0]*(len(X_clients)-n_adversaire) + [1]*n_adversaire

    for fl in range(fl_iterations):
        print('Federated learning iteration: ', fl + 1)
        weights = np.array(model_init[1], dtype='object') * 0  # initialize empty weights

        weights_list = []
        for i in range(len(X_clients)):
            if fl == 0:
                if adversaire[i]:
                    model_client = net_adversary(X_clients[i], y_clients[i], X_test, y_test, epochs=epochs_FL, weights=False, verbose=0,dataset=dataset)
                else :
                    model_client = net(X_clients[i], y_clients[i], X_test, y_test, epochs=epochs_FL, weights=False, verbose=0,dataset=dataset)
            else:
                if adversaire[i]:
                    model_client = net_adversary(X_clients[i], y_clients[i], X_test, y_test, epochs=epochs_FL, weights=FL_weights, verbose=0,dataset=dataset)
                else :
                    model_client = net(X_clients[i], y_clients[i], X_test, y_test, epochs=epochs_FL, weights=FL_weights, verbose=0,dataset=dataset)
            weights_list.append(np.array(model_client[1], dtype='object'))

        FL_weights = aggregator_mean(weights_list)

        test = np.array(FL_weights, dtype='object')

        # set aggregated weights
        model_FL = model_client[0]
        model_FL.set_weights(FL_weights)

        # compute predictions using aggregated weights
        y_pred = np.argmax(model_FL.predict(X_test), axis=1)
        y_test_argmax = np.argmax(y_test, axis=1)
        acc = np.mean(y_pred == y_test_argmax)
        print("Federated Accuracy: ", acc)
        acc_fl.append(acc)

    return acc_fl