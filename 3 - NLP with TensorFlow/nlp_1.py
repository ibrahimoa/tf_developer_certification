from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences #To use padding functions

# String tokenization
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'He loves his cat',
    'She loves his dog',
    'Do you think my dog is amazing?'
]

test_data = [
    'i really love my dog',
    'my dog loves my manatee',
    'my dog loves my banana haha'
]
tokenizer  =Tokenizer(num_words=100, oov_token="<OOV>") #OOV = Out Of Vocabulay
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
test_seq = tokenizer.texts_to_sequences(test_data)

padded = pad_sequences(sequences) #The sequence now becomes a matrix,
# and each row has the same length (putting zeros in the beginning if it's necessary)
test_padded = pad_sequences(test_seq, padding='post', maxlen=5, truncating='post') #This adds the padding at the end
# and also makes the max length of each row become 5 -> We may lose information. The default option
# is set to be 'pre' but we can change that.

print("\nWord Index = ", word_index) #We assign an integer value to each word present in the sencentes
print("\nSequences = ", sequences) #Tokens replacing the words
print("\nTest Sequences = ", test_seq)
print("········································································")
print("\nPadded Sequences = ", padded)
print("\nTest Padded Sequences = " ,test_padded)
print("\nPadded shape = ", padded.shape)