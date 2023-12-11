import numpy as np
from aggregators import aggregator_mean
from train import net,net_backdoor

def federated(X_clients, y_clients, X_test, y_test, fl_iterations, n_backdoor=0, model_name=None):
    '''This function performs the Federated Learning'''
    model_init = net(X_clients[0], y_clients[0], X_test, y_test, epochs=1, weights=False, verbose=0)
    acc_fl = []

    backdoor = [0] * (len(X_clients) - n_backdoor) + [1] * n_backdoor

    for fl in range(fl_iterations):
        print('Federated learning iteration: ', fl + 1)
        weights = np.array(model_init[1], dtype='object') * 0  # initialize empty weights

        weights_list = []
        for i in range(len(X_clients)):
            if fl == 0:
                if backdoor[i]:
                    model_client = net_backdoor(X_clients[i], y_clients[i], X_test, y_test, epochs=1, weights=False, verbose=0)
                else :
                    model_client = net(X_clients[i], y_clients[i], X_test, y_test, epochs=1, weights=False, verbose=0)
            else:
                if backdoor[i]:
                    model_client = net_backdoor(X_clients[i], y_clients[i], X_test, y_test, epochs=1, weights=FL_weights, verbose=0)
                else :
                    model_client = net(X_clients[i], y_clients[i], X_test, y_test, epochs=1, weights=FL_weights, verbose=0)
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

        if model_name is None:
            pass
        else :
            model_FL.save_weights(f'models/{model_name}')

    return acc_fl, model_FL, FL_weights