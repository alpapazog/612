import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models import build_logistic_model, build_ffnn_model, build_cnn_model

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Load data
vocab_size = 5000
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Train logistic
log_model = build_logistic_model(vocab_size, maxlen)
log_model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
_, acc = log_model.evaluate(x_test, y_test)
print(f"Logistic Accuracy: {acc * 100:.2f}%")

# FFNN
ffnn = build_ffnn_model(vocab_size, maxlen)
ffnn.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
_, acc = ffnn.evaluate(x_test, y_test)
print(f"FFNN Accuracy: {acc * 100:.2f}%")

# CNN
cnn = build_cnn_model(vocab_size, maxlen)
cnn.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
_, acc = cnn.evaluate(x_test, y_test)
print(f"CNN Accuracy: {acc * 100:.2f}%")
