from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D

def build_logistic_model(vocab_size=5000, maxlen=200):
    model = Sequential([
        Embedding(vocab_size, 32, input_length=maxlen),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_ffnn_model(vocab_size=5000, maxlen=200):
    model = Sequential([
        Embedding(vocab_size, 32, input_length=maxlen),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(vocab_size=5000, maxlen=200):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=maxlen),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
