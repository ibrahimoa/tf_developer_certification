import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku

tokenizer = Tokenizer()
data = open("irish-lyrics-eof.txt").read()
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus) # Creates de dictionary (word = key, value = token of the word)
total_words = len(tokenizer.word_index) + 1 # +1 to consider OOV words

input_sequences = [] # Training data
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0] # Line of text to a list of tokens representing this text
    for i in range(1, len(token_list)): # Iterate over the list of tokens. Generating lists composed by tokens (first 2, then 3, 4, etc.)
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# The last element if each list is our label (Y), the others are our input (X)

xs = input_sequences[:,:-1] # Everything but the last token
labels = input_sequences[:,-1] # Keep the last token
ys = ku.to_categorical(labels, num_classes=total_words) # Convert list to categorical

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150, return_sequences = True))) # Carry content around with them
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01))),
model.add(Dense(total_words, activation='softmax')),
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

seed_text = "I've got a bad feeling about this"
next_words = 10

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
    print(seed_text)