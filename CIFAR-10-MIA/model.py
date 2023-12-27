from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import numpy as np

class Model(Sequential):
    def __init__(self, num_classes=10, **kwargs):
        super(Model, self).__init__(**kwargs)

        self.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
        self.add(Activation('relu'))
        self.add(Conv2D(32, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))

        self.add(Conv2D(64, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(Conv2D(64, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))

        self.add(Flatten())
        self.add(Dense(512))
        self.add(Activation('relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_classes))
        self.add(Activation('softmax'))

    def fit(self, x=None, y=None, **kwargs):
        super(Model, self).fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        prediction = super(Model, self).predict(x, **kwargs)
        """Add noise to the prediction to prevent the MIA
        Change the max of each array to the median of the array
        # Get the highest confidence levels
        maxes = np.max(prediction, axis=1)
        median = np.median(maxes)
        indices = np.argmax(prediction, axis=1)

        for i in range(len(prediction)):
            prediction[i][indices[i]] = median"""

        return prediction