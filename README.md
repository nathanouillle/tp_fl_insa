# TP Federate Learning

You will implement federated learning among multiple participants 
and analyze the impact of a poisoning attack and a membership inference attack, 
as well as their countermeasures.

### Installation  

```shell=bash
git clone https://gitlab.inria.fr/nbrunet/tp-fl-insa.git
cd tp-fl-insa
python -m venv .tp
source .tp/bin/activate
pip install numpy pandas matplotlib tensorflow
```

##### Installation test
Execute the Example/main.py file and verify that everything is working, 
it should run with the MNIST and CIFAR-10 datasets.

```shell=bash
cd Example
python main.py
```

### Arborescence du projet

```
├── Backdoor
│   ├── aggregators.py # file with all the aggregators 
│   │   ├── aggregator_mean() # aggregate the weights with a mean
│   ├── federated_learning.py # does the federated learning loop, send data to clients or adversarial clients, collects weights, call the aggregator,
│   ├── main.py  # defines the variables, interacts with preprocessing.py to collect datas, sends it to federated_learning.py, shows the results
│   ├── preprocessing.py # loads the dataset, splits the dataset
│   │   ├── load_data() return the complete dataset
│   │   ├── data_to_clients_random() # splits the datas in n_clients randomly
│   └── train.py
│   │   ├── net() # defines the model and the training of a client
│   │   ├── net_backdoor() # defines the model and the training of a backdoor client
|
├── CIFAR-10-MIA
│   ├── data # images used or not during the training
│   │   ├── x.npy
│   │   ├── y.npy
│   ├── ground_truth.txt # contains the truth about train and test datas 
│   ├── main.py # inferes on the datas after loading the model, sends the data to the MIA attack, evaluates the MIA return by attack() with the submission and ground_truth
│   ├── MIA.py
│   │   ├── attack() # contains your MIA attack, returns the submission
│   ├── model.py # contains the class Model. The method predict() can be modified to inluence what the model return.
│   ├── models 
│   │   ├── ...
|
├── Example
│   ├── aggregators.py # file with all the aggregators 
│   │   ├── aggregator_mean() # aggregate the weights with a mean
│   ├── federated_learning.py # does the federated learning loop, send data to clients, collects weights, call the aggregator
│   ├── main.py  # defines the variables, interacts with preprocessing.py to collect datas, sends it to federated_learning.py, shows the results
│   ├── preprocessing.py # loads the dataset (MNIST or CIFAR-10), splits the dataset
│   │   ├── load_data() return the complete dataset
│   │   ├── data_to_clients_random() # splits the datas in n_clients randomly    
│   │   ├── data_to_clients_pond() # splits the datas in 5 clients, with 2 classes for each
│   │   ├── data_to_clients_custom() # splits the datas with you custom function
│   └── train.py
│   │   ├── net() # defines the model and the training of a client
|
├── Poison
│   ├── aggregators.py # file with all the aggregators 
│   │   ├── aggregator_mean() # aggregate the weights with a mean
│   │   ├── aggregator_custom() # aggregate the weights with you custom aggregators
│   ├── federated_learning.py # does the federated learning loop, send data to clients or adversarial clients, collects weights, call the aggregator,
│   ├── main.py  # defines the variables, interacts with preprocessing.py to collect datas, sends it to federated_learning.py, shows the results
│   ├── preprocessing.py # loads the dataset (MNIST or CIFAR-10), splits the dataset
│   │   ├── load_data() return the complete dataset
│   │   ├── data_to_clients_random() # splits the datas in n_clients randomly    
│   │   ├── data_to_clients_pond() # splits the datas in 5 clients, with 2 classes for each
│   │   ├── data_to_clients_custom() 
│   └── train.py
│   │   ├── net() # defines the model and the training of a client
│   │   ├── net_adversarial() # defines the model and the training of an adversarial client

├── README.md
```

### Definition of the models

The model for classifying MNIST images is a dense neural network defined sequentially, fully connected, with a rectifier (ReLU) as the activation function.

The model for classifying CIFAR-10 images is a Convolutional Neural Network (CNN) with a rectifier (ReLU) as the activation function.
### Getting Started

Observe the influence of the number of epochs, the number of clients, and the size of the dataset

Use the function ```data_to_clients_pond()``` with 5 clients to see the impact of dataset division. 

Implement the function ```data_to_clients_custom()``` with 5 clients as explained in the comments.


```python
def data_to_clients_custom(X_train,y_train,n_clients=5):
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

    X_clients = []
    y_clients = []

    return X_clients,y_clients


```

### Poisoning 


For the following questions, we will use  ```data_to_clients_random()```, and the MNIST dataset.


In this section, we will set up a poisoning attack. 
An adversarial client will pollute the received model to make it less effective. 
To achieve this, the adversarial client will randomly set the weights of the neurons before sending them back to the aggregation server. 
Modify the number of adversaries and the ```net_adversaire() ```function to make it return random weights.

```python
def net_adversary(X_train, y_train, X_test, y_test, epochs, weights=None, verbose=1,dataset='MNIST',num_classes=10):
    '''Create the Neural Network'''
    
    #[...]
    
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

```
The shape of the weights of the returned model must belong the same
Run the training and verify that the model's performance has indeed degraded.

```shell=bash
cd Poison
python main.py
```

Observe the influence of the number of adversaries compared to the total number of clients.
Another attack than random weights can also be relevant

### Defense against poisoning

In this section, we will assume that the model is poisoned by a Poisoning attack as seen previously,
and we will implement a countermeasure.
It is necessary to use a different form of aggregation less sensitive to extreme values. 
While retaining your poisoning attack from the previous step, 
modify the ```aggregator_custom()``` function to implement a countermeasure.

```python
def aggregator_custom(weights_list):
    ''' Must return the same objects as aggregator_mean'''
    return None
```

Call this function in ```federated_learning.py``` instead of```aggregator_mean()```.

Check that the training is not impacted by the presence of a poisoned adversary. 

With 6 clients, increasing the number of adversaries, highlight the limit of the implemented countermeasure. 

Code a more robust method than the one coded previously.


### Membership Inférence Attack (MIA)

A membership inference attack aims to determine if specific data points were part of a model's training dataset.
We will implement an attack based on the model's accuracy. 
This accuracy will be different for data used or not used during training (higher accuracy for data that the model knows). 
Thus, with a well-chosen threshold,
it will be possible to differentiate data points belonging to the training set from those that do not.

We will now use CIFAR-10 dataset

Using the provided model and the dataset (x.npy and y.npy) consisting of mixed training and test data,
visualize the distribution of the model's confidence levels across all data points using a histogram. Use matplotlib

Then, define a threshold that could separate the training and test data, and implement the MIA


Modify  ```MIA.py```, and ```main.py```, 

```python
def attack(predictions,treshold):
    '''
    Submission is a list composed of 1 (train) and 0 (test), with the same len as predictions (number of data in the
    dataset). It's contains the result of the MIA.

    '''

    submission = [0]*len(predictions)
    return submission
```

Retrieve the model predictions on the data (store them in the variable `y_hat`, for example), 
call the function `attack()`, 
and store the result in the variable `submission`.

```shell=bash
cd CIFAR-10-MIA
python main.py
```

Adjust you attack to improve your score

Another attack different from the MIA with a threshold could also be relevant.

### Defense against MIA

The principle of this countermeasure is to add noise to the model's output to prevent a MIA. 
Modify the `predict()` method of the `Model` class to add noise.

```python
def predict(self, x, **kwargs):
    return super(Model, self).predict(x, **kwargs)
```

Observe the influence of your defense on the effectiveness of the MIA. 
Finally, try to find the best trade-off between defense and utility 
by using the model accuracy to attack accuracy ratio

### Backdoor

We will implement a backdoor attack. 
An adversarial client will try to influence the model during training to make it misclassify a specific portion of the dataset. A backdoor attack does not aim to degrade the model, risking exclusion from training. 
In our case, on CIFAR-10 data, 
you need to ensure that the aggregated model classifies as many cars as possible as airplanes.


Similar to the Poisoning task, you need to modify the `net_backdoor()` function 
and the number of clients applying this backdoor.


```python
def net_backdoor(X_train, y_train, X_test, y_test, epochs, weights=None, verbose=1, num_classes=10):
    '''Create the Neural Network'''

    # [...]

    if type(weights) == np.ndarray:
        model.set_weights(weights)

    # fit model
    history = model.fit(X_train, y_train,batch_size=128, epochs=epochs,validation_data=(X_test, y_test),verbose=0)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_argmax = np.argmax(y_test, axis=1)
    acc = np.mean(y_pred == y_test_argmax)

    if verbose == 1:
        print(f'Accuracy of the model: {acc:.3f}')

    return model, model.get_weights(), history, acc
```

Test the backdoor

```shell=bash
cd Backdoor
python main.py
```

You can use other methods (and combine them) to increase the number of cars classified as airplanes by the aggregated model.


You will need to optimize the absolute value of the ratio between 
the difference in misclassification between a healthy model and a model with your backdoor 
and the difference in precision between a healthy model and a model with your backdoor.

```python
if True : # Switch to True for Defense
    TP = 0
    for pred, truth in zip(y_hat,y):
        if np.argmax(pred) == np.argmax(truth):
            TP += 1

    accu = TP/len(y_hat)

    print(f"Le modèle obtient une précision de : {accu}")

    print(f"score final : {accu/MIA}")
```

Github of the original notebook : https://github.com/oscar-defelice/DeepLearning-lectures

