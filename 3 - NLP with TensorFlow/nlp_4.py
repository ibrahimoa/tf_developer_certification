import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']
train_data = train_data.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))
test_data = test_data.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

tokenizer = info.features['text'].encoder
print(tokenizer.subwords)

sample_string = 'Tensorflow, from basics to mastery'
tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string: {}'.format(tokenized_string))
original_string = tokenizer.decode(tokenized_string)
print('The original string: {}'.format(original_string))

embedding_dim = 64
model = tf.keras.Sequential([
    layers.Embedding(tokenizer.vocab_size, embedding_dim),
    layers.GlobalAveragePooling1D(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
print(model.summary())

num_epochs = 10
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_data,
                    epochs=num_epochs,
                    validation_data=test_data)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val:' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
