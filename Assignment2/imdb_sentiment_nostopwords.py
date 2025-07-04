import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from models import build_logistic_model, build_ffnn_model, build_cnn_model

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Get IMDB + word index
vocab_size = 5000
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
word_index = imdb.get_word_index()
index_word = {i + 3: w for w, i in word_index.items()}
index_word[0] = '<PAD>'
index_word[1] = '<START>'
index_word[2] = '<OOV>'
stop_words = set(stopwords.words('english'))

# Remove stopwords
def clean_text(encoded_review):
    return [w for w in [index_word.get(i, '?') for i in encoded_review] if w not in stop_words and w not in {'<PAD>', '<START>', '<OOV>', '<UNUSED>'}]

train_text = [' '.join(clean_text(seq)) for seq in x_train]
test_text = [' '.join(clean_text(seq)) for seq in x_test]

# Tokenize
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_text)
x_train = tokenizer.texts_to_sequences(train_text)
x_test = tokenizer.texts_to_sequences(test_text)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Train logistic
log_model = build_logistic_model(vocab_size, maxlen)
log_model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
_, acc = log_model.evaluate(x_test, y_test)
print(f"Logistic (No Stopwords) Accuracy: {acc * 100:.2f}%")

# FFNN
ffnn = build_ffnn_model(vocab_size, maxlen)
ffnn.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
_, acc = ffnn.evaluate(x_test, y_test)
print(f"FFNN (No Stopwords) Accuracy: {acc * 100:.2f}%")

# CNN
cnn = build_cnn_model(vocab_size, maxlen)
cnn.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
_, acc = cnn.evaluate(x_test, y_test)
print(f"CNN (No Stopwords) Accuracy: {acc * 100:.2f}%")
