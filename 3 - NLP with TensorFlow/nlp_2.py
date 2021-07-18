import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Embeddings in NLP

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# train_data[i] = [sentence, label]
for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

# test_data[i] = [sentence, label]
for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

print(len(training_labels_final))
print(len(testing_labels_final))

vocab_size: int = 10000
embedding_dim: int = 16
max_length: int = 120
trunc_type: str = 'post'
oov_tok: str = '<OOV>'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
trainning_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(trainning_sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

# Often words in a sentence that have similar meanings are close to each other. So in a movie review, maybe it say that
# the movie was dull and boring, or it might say that it was fun and exciting. We then pick a vector in a higher-dimensional
# space (16 for example) and words that are found together are given similar vectors. Then over time, words begin to cluster
# together. The meaning of the words can come from the labeling of the data set. So in this case, negative review and the
# words dull and boring show up a lot in the negative review so that they have similar sentiments.

model = tf.keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),  # The key to test sentiment analysis in tf.
    layers.GlobalAveragePooling1D(),  # Averages across the vector to flatten it out (we can also use flatten)
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
num_epochs: int = 20
model.fit(training_padded,
          training_labels_final,
          epochs=num_epochs,
          validation_data=(testing_padded, testing_labels_final))

e = model.layers[0]  # Embedding layer
weights = e.get_weights()[0]
print(weights.shape)  # shape: (vocab_size, embedding_dim) = (10000, 16)

# Before we had numbers representing words, now we get the words representing the numbers
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

import io

out_v = io.open('tsv\\vecs.tsv', 'w+', encoding='utf-8')
out_m = io.open('tsv\\meta.tsv', 'w+', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")  # Coefficient of each dimension on the vector for this word
out_v.close()
out_m.close()
