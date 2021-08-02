from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
from sklearn.model_selection import train_test_split
from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_historys
import requests
import os.path
import pandas as pd
import random

NLP_DATA_PATH = 'datasets/nlp'
ZIP_FILE_PATH = os.path.join(NLP_DATA_PATH, 'nlp_getting_started.zip')
TRAIN_DATA_PATH = os.path.join(NLP_DATA_PATH, 'train.csv')
TEST_DATA_PATH = os.path.join(NLP_DATA_PATH, 'test.csv')

if os.path.isfile(ZIP_FILE_PATH):
    print('The data is already in your computer!')
else:
    DATA_URL = "https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip"
    print('Downloading the data ...')
    r = requests.get(DATA_URL)
    with open(ZIP_FILE_PATH, 'wb') as f:
        f.write(r.content)

if os.path.isfile(TRAIN_DATA_PATH) and os.path.isfile(TEST_DATA_PATH):
    print('Data already unzipped!')
else:
    print('Unzipping data ...')
    unzip_data(NLP_DATA_PATH)

# turn .csv files into pandas DataFrames's
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
# train_df.head()

# shuffle the data
train_df_shuffled = train_df.sample(frac=1, random_state=42)
# train_df_shuffled.head()

# how many examples of each class ?
print(train_df_shuffled.target.value_counts())
# In this case we have a binary classification problem where:
# 1 = a real disaster Tweet
# 0 = not a real disaster Tweet

# How many samples total?
print(f"Total training samples: {len(train_df_shuffled)}")
print(f"Total test samples: {len(test_df)}")
print(f"Total samples: {len(train_df_shuffled) + len(test_df)}")

# now let's visualize some random training examples
random_index = random.randint(0, len(train_df_shuffled)-5)
print('\n---')
for row in train_df_shuffled[["text", "target"]][random_index:random_index+5].itertuples():
    _, text, target = row
    print(f"Target: {target}", "(real disaster)" if target >
          0 else "(not real disaster)")
    print(f"Text: {text}\n---")

# split data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1,
                                                                            random_state=42)

# converting text into numbers
# we have two options, using Tokenization or Embeddings.
# text_vectorizer = TextVectorization(max_tokens=None,  # how many words in the vocabulary.
#                                     standardize="lower_and_strip_punctuation",  # how to process text.
#                                     split="whitespace",  # how to split tokens.
#                                     ngrams=None,  # create groups of n-words.
#                                     output_mode="int",  # how to map tokens to numbers.
#                                     output_sequence_length=None,  # how long should the output sequence of tokens be.
#                                     pad_to_max_tokens=True)

# find the average number of tokens (words) in training Tweets
print(round(sum([len(i.split()) for i in train_sentences])/len(train_sentences)))

# Setup text vectorization variables
max_vocab_length: int = 10000  # max number of words to have in our vocabulary
max_length: int = 15  # max length our sequences will be (e.g. how many words from a Tweet does our model see?)

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)
text_vectorizer.adapt(train_sentences)

sample_sentence = "There's a flood in my street!"
print(text_vectorizer([sample_sentence]))

# Get the unique words in the vocabulary
words_in_vocab = text_vectorizer.get_vocabulary()
top_5_words = words_in_vocab[:5] # most common tokens (notice the [UNK] token for "unknown" words)
bottom_5_words = words_in_vocab[-5:] # least common tokens
print(f"Number of words in vocab: {len(words_in_vocab)}")
print(f"Top 5 most common words: {top_5_words}")
print(f"Bottom 5 least common words: {bottom_5_words}")

# creating an Embedding using an Embedding Layer

# The powerful thing about an embedding is it can be learned during
# training. This means rather than just being static (e.g. 1 = I,
# 2 = love, 3 = TensorFlow), a word's numeric representation can be
# improved as a model goes through data samples.

from tensorflow.keras import layers

embedding = layers.Embedding(input_dim=max_vocab_length,
output_dim=128,
embeddings_initializer="uniform",
input_length=max_length)

random_sentence = random.choice(train_sentences)
print(f"Original text:\n{random_sentence}\
    \n\nEmbedded version:")
sample_embed = embedding(text_vectorizer([random_sentence]))
print(sample_embed)
