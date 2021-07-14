import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.25):
            print('\nLoss is below 25 per cent so cancelling training')
            self.model.stop_training = True
callback25 =  myCallback()
fashion_mnist = keras.datasets.fashion_mnist
#load_data returns four lists:
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
train_labels = train_labels / 255.0
test_images = test_images / 255.0
test_labels = test_labels / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #Images size is 28x28. This layer tunrs this 28x28 array into a linear array
    keras.layers.Dense(128, activation=tf.nn.relu), #Hidden layer -> 128 neurons
    keras.layers.Dense(10, activation=tf.nn.softmax)] #10 neurons because we have 10 classes
)
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy']) #stochastic gradient descent
model.fit(train_images, train_labels, epochs=100, callbacks=[callback25])
print(model.evaluate(test_images, test_labels))