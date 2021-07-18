import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 1000  # 10000
embedding_dim = 8  # 16
max_length = 32  # 32
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_size = 20000

# Loss is the "confidence in the prediction"

with open("sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

print(f'Total data: {len(sentences)}')
print(f'\tTraining data: {training_size}')
print(f'\tTesting data: {len(sentences) - training_size}')

training_sentences = sentences[:training_size]  # From 0 to training_size
testing_sentences = sentences[training_size:]  # The rest
training_labels = np.array(labels[:training_size])
testing_labels = np.array(labels[training_size:])

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = np.array(pad_sequences(training_sequences, maxlen=max_length,
                                         padding=padding_type, truncating=trunc_type))

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = np.array(pad_sequences(testing_sequences, maxlen=max_length,
                                        padding=padding_type, truncating=trunc_type))

from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),  # The key to test sentiment analysis in tf.
    layers.GlobalAveragePooling1D(),  # Averages across the vector to flatten it out (we can also use flatten)
    layers.Dropout(0.2),
    layers.Dense(24, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
num_epochs = 30
history = model.fit(training_padded,
                    training_labels,
                    epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels),
                    verbose=2)

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
