import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

from model import Model
from MIA import attack

num_classes = 10
treshold = 0.9579635262489319 # Mean confidence level of correct classifications

X = np.load(f'data/x.npy')
y = np.load(f'data/y.npy')

model = Model(num_classes=num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('models/CIFAR-1.4')

y_hat = model.predict(X)

submission = attack(y_hat,treshold)

with open('ground_truth.txt') as file:
    ground_truth = [int(line.strip()) for line in file]

with open('submission.txt','w') as file:
    for bool in submission:
        file.write(str(bool)+'\n')

# Scoring the MIA
s = 0
for a, b in zip(ground_truth, submission):
    if int(b) == 0 or int(b) == 1:
        s += abs(int(a) - int(b))
    else:
        s = len(ground_truth)
        break
MIA = (len(ground_truth) - s) / len(ground_truth)
print("-------------------------------------")
print(f"MIA accuracy : {MIA}")
print("-------------------------------------")

# Scoring Defense against MIA

if True : # Switch to True for Defense
    TP = 0
    for pred, truth in zip(y_hat,y):
        if np.argmax(pred) == np.argmax(truth):
            TP += 1

    accu = TP/len(y_hat)

    print(f"model accuracy : {accu}")

    print(f"score : {accu/MIA}")