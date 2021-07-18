import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Recurrent Neural Networks RNN
# LSTM : Long short-term memory (RNN architecture)
# Embeddings in NLP

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

# Often words in a sentence that have similar meanings are close to each other. So in a movie review, maybe it say that
# the movie was dull and boring, or it might say that it was fun and exciting. We then pick a vector in a higher-dimensional
# space (16 for example) and words that are found together are given similar vectors. Then over time, words begin to cluster
# together. The meaning of the words can come from the labeling of the data set. So in this case, negative review and the
# words dull and boring show up a lot in the negative review so that they have similar sentiments.

# Gated Recurrent Units (GRUs) are a gating mechanism in recurrent neural networks, introduced in 2014 by Kyunghyun Cho.
# The GRU is like a long short-term memory (LSTM) with a forget gate, but has fewer parameters than LSTM, as it lacks an
# output gate. GRU's performance on certain tasks of polyphonic music modeling, speech signal modeling and natural language
# processing was found to be similar to that of LSTM.[4][5] GRUs have been shown to exhibit better performance on certain
# smaller and less frequent datasets.

model = tf.keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),  # The key to test sentiment analysis in tf.
    layers.Dropout(0.2),
    layers.Bidirectional(layers.GRU(32)),  # GRU is a type of architecture RNN
    # layers.Conv1D(128, 5, activation='relu'),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
num_epochs = 20
history = model.fit(training_padded,
                    training_labels_final,
                    epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels_final))

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
