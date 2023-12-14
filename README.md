# TP Federate Learning

Dans ce TP, vous allez mettre en pratique un apprentissage fédéré entre plusieurs
participants et analyser l’impact d’une attaque de poisoning et d’une
attaque d’appartenance ainsi que leurs contre-mesures.

### Installation  

```shell=bash
git clone https://gitlab.inria.fr/nbrunet/tp-fl.git
cd tp-fl
python -m venv .tp
source .tp/bin/activate
pip install numpy pandas matplotlib tensorflow
```

##### Test de l'installation :
Exécuter le fichier Example/main.py pour vérifier que tout fonctionne. Il doit tourner avec les données MNIST et CIFAR-10.

```shell=bash
python Example/main.py
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

### Définition des models

Le modèle pour classifier les images MNIST est un réseau de neurones dense, définit de manière séquentielle, fully connected, avec un rectifier (relu) comme fonction d'activation.
Le modèle pour classifier les images CIFAR-10 est un CNN , avec un rectifier (relu) comme fonction d'activation.
### Prise en main

Prendre la main sur le fonctionnement et sur les différents fichiers du TP. 
Constater ensuite l'influence du nombre d'epochs, du nombre de clients, de la taille du jeu de donnée. 
Essayer la fonction ```data_to_clients_pond()``` avec 5 clients pour voir l'influence de la division du dataset.
Implémenter la fonction ```data_to_clients_custom()``` avec 5 clients comme expliqué en commentaire.

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


On utilisera maintenant ```data_to_clients_random()```, et le jeu de donnée MNIST pour les prochaines questions.

Dans cette section, nous allons mettre en place une attaque de poisoning.
Un client adversaire va polluer le modèle qu’il reçoit afin de le rendre moins
performant. Pour cela, le client adversaire va définir aléatoirement les poids des
neurones avant de les renvoyer au serveur d’agrégation.
Modifier le nombre d'adversaires ainsi que la fonction ```net_adversaire() ``` pour qu'elle renvoie des poids aléatoires. 

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

La forme des poids du modèle renvoyé ne doit pas être modifiée.
Lancer l'entraînement et vérifier que les performances du modèle ont bien été dégradées.

```shell=bash
python Poison/main.py
```

Constater ensuite l'influence du nombre d'adversaires par rapport au nombre de clients total.
Une autre attaque que des poids aléatoires peut aussi être pertinente.

### Defense against poisoning

Dans cette section, nous considérerons que le modèle est empoisonné par une attaque de type Poisoning vue précédemment 
et nous allons implémenter une contre-mesure.
Il faut utiliser une autre forme d'agrégation moins sensible aux valeurs extrêmes. 
En conservant votre attaque de poisoning de l'étape précédente, modifier la fonction ```aggregator_custom()``` pour implémenter une contre-mesure.

```python
def aggregator_custom(weights_list):
    ''' Must return the same objects as aggregator_mean'''
    return None
```

Appeler cette fonction dans ```federated_learning.py``` à la place d'```aggregator_mean()```.

Vérifier que l'entraînement n'est pas impacté par la présence d'un adversaire empoisonné. 
Avec 6 clients, en augmentant le nombre d'adversaires, mettre en évidence la limite de la contre-mesure implémentée.

Coder une méthode plus robuste que celle codée précédemment. 



### Membership Inférence Attack (MIA)

Une attaque d’appartenance (ou de Membership) vise à savoir si des données 
spécifiques faisaient partie du jeu de données d'entraînement d’un modèle.
Nous allons implémenter une attaque basée sur la précision du modèle. Celle-ci sera
différente pour des données utilisées ou non pour l'entraînement (la précision sera plus
haute pour des données que le modèle connaît). Ainsi avec un seuil bien
choisi, il sera possible de différencier des données faisant partie du jeu
d'entraînement, ou non.

Cette section sera fait sur des données du dataset CIFAR-10. 
En utilisant le modèle fourni et le jeu de donnée (x.npy et y.npy) composé de données d'entrainement et de données de test mélangées,
le but est d'implémenter une MIA et de l'évaluer. 
Avant de coder une attaque de MIA, visualiser à l'aide d'un histogramme la distribution des niveaux de confiance du modèle sur l'ensemble des données.
Définir ensuite un seuil qui pourrait séparer les données d'entraînement et de test.

En modifiant ```MIA.py```, et ```main.py```, 

```python
def attack(predictions,treshold):
    '''
    Submission is a list composed of 1 (train) and 0 (test), with the same len as predictions (number of data in the
    dataset). It's contains the result of the MIA.

    '''

    submission = [0]*len(predictions)
    return submission
```

Récupérer les predictions du modèle sur les données (dans la variable ```y_hat``` par exemple), appeler la fonction ```attack()```, 
dont le résultat sera stocké dans la variable ```submission```. 

```shell=bash
python CIFAR-10-MIA/main.py
```

Ajuster votre attaque pour améliorer votre MIA.

Une autre attaque différente de la MIA avec seuil peut aussi être pertinente.

### Defense against MIA

Le principe de cette contre-mesure est d'ajouter du bruit dans le retour modèle pour éviter une MIA.
Modifier la méthode ```predict()``` de la classe ```Model``` afin d'ajouter du bruit.

```python
def predict(self, x, **kwargs):
    return super(Model, self).predict(x, **kwargs)
```

Constater l'influence de votre défense sur l'efficacité de la MIA.
Enfin, essayer de trouver le meilleur compromis defense/utilité grâce au rapport précision du modèle sur précision de la MIA.

### Backdoor

Dans cette section, nous allons mettre en place une attaque de type backdoor.
Un client adversaire va essayer d'influencer le modèle lors de l'entraînement
afin qu'il fasse une erreur de classification sur une partie précise du dataset.
Une backdoor ne vise pas à détériorer la modèle au risque d'être exclue de l'entraînement.
Dans notre cas, sur des données CIFAR-10, il faudra faire en sorte que le modèle agrégé classe le plus possible de voiture en avion. 

De la même manière que pour le poisoning, il faut modifier la fonction ```net_backdoor()```, 
et le nombre de clients qui appliquent cette backdoor. 


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

Tester la backdoor

```shell=bash
python Backdoor/main.py
```

Vous pouvez utiliser différentes méthodes pour augmenter le nombre de voitures classées comme des avions par le modèle agrégé. 



Il faudra optimiser la valeur absolue du rapport entre la différence de mauvaise classification entre un modèle sain et un modèle avec votre backdoor,
et la différence de précision entre un modèle sain et un modèle avec votre backdoor. 

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