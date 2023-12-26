import numpy as np
import sys
def aggregator_mean(weights_list):
    return np.mean(weights_list,axis=0)

def aggregator_custom(weights_list):
    '''
    :param weights_list:
    :return: np.array(_______ ,dtype='object')

    Must return the same objects as aggregator_mean and aggregator_median
    '''

    # Commun to all layers
    num_clients = len(weights_list)

    # Layer 0: 784,1000
    row_length_0 = len(weights_list[0][0])
    column_length_0 = len(weights_list[0][0][0])

    # Layer 1: 1000,
    column_length_1 = len(weights_list[0][1])

    # Layer 2: 1000,1000
    row_length_2 = len(weights_list[0][2])
    column_length_2 = len(weights_list[0][2][0])

    # Layer 3: 1000,
    column_length_3 = len(weights_list[0][3])

    # Layer 4: 1000,10
    row_length_4 = len(weights_list[0][4])
    column_length_4 = len(weights_list[0][4][0])

    # Layer 5: 10,
    column_length_5 = len(weights_list[0][5])

    # Structures pour stocker (Rows)/Columns/Client
    """J'ai choisi cette structure car 
        1. Elle est plus facile à manipuler que la structure de base
        2. La dernière dimension est la liste des poids fournis par les clients, pour chaque matching neurone : Extrêmement pratique pour calculer la médiane"""
    layer_0 = np.zeros((row_length_0,column_length_0,num_clients))
    layer_1 = np.zeros((column_length_1,num_clients))
    layer_2 = np.zeros((row_length_2,column_length_2,num_clients))
    layer_3 = np.zeros((column_length_3,num_clients))
    layer_4 = np.zeros((row_length_4,column_length_4,num_clients))
    layer_5 = np.zeros((column_length_5,num_clients))

    # On remplit le tableau layer 0
    print("Computing layer 0...")
    for row in range(row_length_0):
        for column in range(column_length_0):
            for client in range(num_clients):
                layer_0[row][column][client] = weights_list[client][0][row][column]
            layer_0[row][column] = aggregator_median(layer_0[row][column],num_clients)
            for client in range(num_clients):
                weights_list[client][0][row][column] = layer_0[row][column][client]
            

    # On remplit le tableau layer 1
    print("Computing layer 1...")
    for column in range(column_length_1):
        for client in range(num_clients):
            layer_1[column][client] = weights_list[client][1][column]
        layer_1[column] = aggregator_median(layer_1[column],num_clients)
        for client in range(num_clients):
            weights_list[client][1][column] = layer_1[column][client]


    # On remplit le tableau layer 2
    print("Computing layer 2...")
    for row in range(row_length_2):
        for column in range(column_length_2):
            for client in range(num_clients):
                layer_2[row][column][client] = weights_list[client][2][row][column]
            layer_2[row][column] = aggregator_median(layer_2[row][column],num_clients)
            for client in range(num_clients):
                weights_list[client][2][row][column] = layer_2[row][column][client]


    # On remplit le tableau layer 3
    print("Computing layer 3...")
    for column in range(column_length_3):
        for client in range(num_clients):
            layer_3[column][client] = weights_list[client][3][column]
        layer_3[column] = aggregator_median(layer_3[column],num_clients)
        for client in range(num_clients):
            weights_list[client][3][column] = layer_3[column][client]


    # On remplit le tableau layer 4
    print("Computing layer 4...")
    for row in range(row_length_4):
        for column in range(column_length_4):
            for client in range(num_clients):
                layer_4[row][column][client] = weights_list[client][4][row][column]
            layer_4[row][column] = aggregator_median(layer_4[row][column],num_clients)
            for client in range(num_clients):
                weights_list[client][4][row][column] = layer_4[row][column][client]

    # On remplit le tableau layer 5
    print("Computing layer 5...")
    for column in range(column_length_5):
        for client in range(num_clients):
            layer_5[column][client] = weights_list[client][5][column]
        layer_5[column] = aggregator_median(layer_5[column],num_clients)
        for client in range(num_clients):
            weights_list[client][5][column] = layer_5[column][client]


    """On calcule la médiane et l'écart type pour chaque neurone
    Si une valeur est un outlier, on la remplace par la médiane"""
    """
    # Layer 0
    print("Calculating median and std dev for layer 0")
    for row in range(row_length_0):
        for column in range(column_length_0):
            list_weights = layer_0[row][column]
            median = np.median(list_weights)
            std_dev = np.std(list_weights)
            for client in range(num_clients):
                if(np.abs(layer_0[row][column][client] - median) > 2 * std_dev):
                    layer_0[row][column][client] = median
    
    # Layer 1
    print("Calculating median and std dev for layer 1")
    for column in range(column_length_1):
        list_weights = layer_1[column]
        median = np.median(list_weights)
        std_dev = np.std(list_weights)
        for client in range(num_clients):
            if(np.abs(layer_1[column][client] - median) > 2 * std_dev):
                layer_1[column][client] = median
    
    # Layer 2
    print("Calculating median and std dev for layer 2")
    for row in range(row_length_2):
        for column in range(column_length_2):
            list_weights = layer_2[row][column]
            median = np.median(list_weights)
            std_dev = np.std(list_weights)
            for client in range(num_clients):
                if(np.abs(layer_2[row][column][client] - median) > 2 * std_dev):
                    layer_2[row][column][client] = median

    # Layer 3
    print("Calculating median and std dev for layer 3")
    for column in range(column_length_3):
        list_weights = layer_3[column]
        median = np.median(list_weights)
        std_dev = np.std(list_weights)
        for client in range(num_clients):
            if(np.abs(layer_3[column][client] - median) > 2 * std_dev):
                layer_3[column][client] = median

    # Layer 4
    print("Calculating median and std dev for layer 4")
    for row in range(row_length_4):
        for column in range(column_length_4):
            list_weights = layer_4[row][column]
            median = np.median(list_weights)
            std_dev = np.std(list_weights)
            for client in range(num_clients):
                if(np.abs(layer_4[row][column][client] - median) > 2 * std_dev):
                    layer_4[row][column][client] = median

    # Layer 5
    print("Calculating median and std dev for layer 5")
    for column in range(column_length_5):
        list_weights = layer_5[column]
        median = np.median(list_weights)
        std_dev = np.std(list_weights)
        for client in range(num_clients):
            if(np.abs(layer_5[column][client] - median) > 2 * std_dev):
                layer_5[column][client] = median

    # On doit remettre les poids dans la structure de base
    # Layer 0
    print("Putting back weights in layer 0")
    for row in range(row_length_0):
        for column in range(column_length_0):
            for client in range(num_clients):
                weights_list[client][0][row][column] = layer_0[row][column][client]

    # Layer 1
    print("Putting back weights in layer 1")
    for column in range(column_length_1):
        for client in range(num_clients):
            weights_list[client][1][column] = layer_1[column][client]

    # Layer 2
    print("Putting back weights in layer 2")
    for row in range(row_length_2):
        for column in range(column_length_2):
            for client in range(num_clients):
                weights_list[client][2][row][column] = layer_2[row][column][client]

    # Layer 3
    print("Putting back weights in layer 3")
    for column in range(column_length_3):
        for client in range(num_clients):
            weights_list[client][3][column] = layer_3[column][client]

    # Layer 4
    print("Putting back weights in layer 4")
    for row in range(row_length_4):
        for column in range(column_length_4):
            for client in range(num_clients):
                weights_list[client][4][row][column] = layer_4[row][column][client]

    # Layer 5
    print("Putting back weights in layer 5")
    for column in range(column_length_5):
        for client in range(num_clients):
            weights_list[client][5][column] = layer_5[column][client]"""
    
    
    return np.mean(weights_list,axis=0)
    
    
def aggregator_median(list_weights, num_clients, threshold=2):
    median = np.median(list_weights)
    std_dev = np.std(list_weights)
    for client in range(num_clients):
        if(np.abs(list_weights[client] - median) > threshold * std_dev):
            list_weights[client] = median
    return list_weights