# TP Federate Learning

Dans ce TP, vous allez mettre en pratique un apprentissage fédéré entre plusieurs
participants et analyser l’impact d’une attaque de poisoning et d’une
attaque d’appartenance ainsi que leurs contre-mesures.

### Installation  

Cloner le repository du TP : ```git clone https://gitlab.inria.fr/nbrunet/tp-fl-insa.git```

Il vous faudra ensuite installer tensorflow et numpy sur votre machine.

##### Test de l'installation :
Exécuter le fichier Poison/main.py pour vérifier que tout fonctionne. Il doit tourner avec les données MNIST et CIFAR-10.

### Prise en main

Prendre la main sur le fonctionnement et sur les différents fichiers du TP. 
Constater ensuite l'influence du nombre d'epochs, du nombre de clients, de la taille du jeu de donnée. 
Essayer la fonction ```data_to_clients_pond()``` avec 5 clients pour voir l'influence de la division du dataset.
Implémenter la fonction ```data_to_clients_custom()``` avec 5 clients comme expliqué en commentaire.

### Poisoning 

On utilisera maintenant ```data_to_clients_random()```, et le jeu de donnée MNIST pour les prochaines questions.

Dans cette section, nous allons mettre en place une attaque de poisoning.
Un client adversaire va polluer le modèle qu’il reçoit afin de le rendre moins
performant. Pour cela, le client adversaire va définir aléatoirement les poids des
neurones avant de les renvoyer au serveur d’agrégation.
Modifier le nombre d'adversaires ainsi que la fonction ```net_adversaire() ``` pour qu'elle renvoie des poids aléatoires. 
La forme des poids du modèle renvoyé ne doit pas être modifiée.
Lancer l'entraînement et vérifier que les performances du modèle ont bien été dégradées.
Constater ensuite l'influence du nombre d'adversaires par rapport au nombre de clients total.
Une autre attaque que des poids aléatoires peut aussi être pertinente.

### Defense against poisoning

Dans cette section, nous considérerons que le modèle est empoisonné par une attaque de type Poisoning vue précédemment 
et nous allons implémenter une contre-mesure.
Il faut utiliser une autre forme d'agrégation moins sensible aux valeurs extrêmes. 
En conservant votre attaque de poisoning de l'étape précédente, modifier la fonction ```aggregator_custom()``` pour implémenter une contre-mesure.
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
récupérer les predictions du modèle sur les données (dans la variable ```y_hat``` par exemple), appeler la fonction ```attack()```, 
dont le résultat sera stocké dans la variable ```submission```. 
Ajuster votre attaque pour améliorer votre MIA.

Une autre attaque différente de la MIA avec seuil peut aussi être pertinente.

### Defense against MIA

Le principe de cette contre-mesure est d'ajouter du bruit dans le retour modèle pour éviter une MIA.
Modifier la méthode ```predict()``` de la classe ```Model``` afin d'ajouter du bruit.
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
Vous pouvez utiliser différentes méthodes pour augmenter le nombre de voitures classées comme des avions par le modèle agrégé. 

Il faudra optimiser la valeur absolue du rapport entre la différence de mauvaise classification entre un modèle sain et un modèle avec votre backdoor,
et la différence de précision entre un modèle sain et un modèle avec votre backdoor. 