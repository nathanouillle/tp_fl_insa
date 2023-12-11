import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aggregators import aggregator_mean
from preprocessing import load_data,data_to_clients_random
from train import net,net_backdoor,create_model
from federated_learning import federated

n_clients = 5
n_backdoor = 2
epochs = 2
n_adversary = 1

# load the dataset
(X_train, y_train, X_test, y_test,X_test_car,y_test_car) = load_data()

# distributed data to clients
X_clients, y_clients = data_to_clients_random(X_train, y_train,n_clients=n_clients)

# federated learning
# federated_learning, model, FL_weights = federated(X_clients, y_clients, X_test, y_test, fl_iterations=epochs, n_backdoor=0,model_name='FL-4')

model = create_model(num_classes=10,model_name='models/FL-4')
federated_learning_backdoor,model_backdoor, FL_weights_backdoor = federated(X_clients, y_clients, X_test, y_test, fl_iterations=epochs, n_backdoor=n_backdoor)

accu,accu_backdoor = 0,0
for result1,result2 in zip(model.predict(X_test_car),model_backdoor.predict(X_test_car)):
    if np.argmax(result1) == 0:
        accu +=1
    if np.argmax(result2) == 0:
        accu_backdoor +=1

accu,accu_backdoor = accu / len(y_test_car), accu_backdoor / len(y_test_car)

loss  = model.evaluate(X_train,y_train)
loss_backdoor = model_backdoor.evaluate(X_train,y_train)


result = abs((accu_backdoor-accu)/(loss[1]-loss_backdoor[1]))

print(f"Pourcentage de voitures classifiées comme des avions par le model sans backdoor : {accu}\n")
print(f"Pourcentage de voitures classifiées comme des avions par le model avec backdoor : {accu_backdoor}")

print(result)