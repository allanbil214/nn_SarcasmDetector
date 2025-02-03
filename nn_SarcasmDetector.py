import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request

# Function to download the dataset, preprocess it, and build/train a model
def solution_model():
    # URL of the dataset
    url = 'http://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    # Download the JSON dataset from the URL and save it locally
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # Set parameters for the model and preprocessing
    vocab_size = 1000  # Maximum number of words to consider in the vocabulary
    embedding_dim = 16  # Dimensionality of the embedding vector
    max_length = 120  # Maximum length of input sequences (padded or truncated)
    trunc_type = 'post'  # Type of truncation ('post' means truncate from the end)
    padding_type = 'post'  # Type of padding ('post' means padding at the end)
    oov_tok = "<OOV>"  # Token to represent out-of-vocabulary words
    training_size = 20000  # Number of training samples

    sentences = []  # List to hold the sentences
    labels = []  # List to hold the sarcasm labels (1 for sarcastic, 0 for not sarcastic)
    
    # Load the dataset from the downloaded JSON file
    with open("sarcasm.json", 'r') as file:
        datastore = json.load(file)
        
        # Extract sentences and corresponding labels
        for item in datastore:
            sentences.append(item['headline'])
            labels.append(item['is_sarcastic'])

    # Convert sentences and labels to numpy arrays for easier processing
    sentences = np.array(sentences)
    labels = np.array(labels)

    # Split the dataset into training and validation sets
    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]
    val_sentences = sentences[training_size:]
    val_labels = labels[training_size:]

    # Initialize the Tokenizer to process the text data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)  # Fit the tokenizer on the training sentences

    # Convert text data to sequences of integers (words are mapped to integer indices)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)

    # Pad the sequences to ensure they are of equal length
    train_padded = pad_sequences(train_sequences, maxlen=max_length, 
                                padding=padding_type, truncating=trunc_type)
    val_padded = pad_sequences(val_sequences, maxlen=max_length,
                              padding=padding_type, truncating=trunc_type)

    # Build the model architecture
    model = tf.keras.Sequential([
        # Embedding layer to convert integer sequences to dense vectors
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        
        # Bidirectional LSTM layer with 32 units, returning sequences for the next layer
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        
        # Another Bidirectional LSTM layer with 16 units
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        
        # Dense layer with 24 units and ReLU activation
        tf.keras.layers.Dense(24, activation='relu'),
        
        # Dropout layer to prevent overfitting by randomly setting input units to 0
        tf.keras.layers.Dropout(0.5),
        
        # Output layer with a sigmoid activation function for binary classification
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model with binary crossentropy loss, Adam optimizer, and accuracy metric
    model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

    # Train the model using the training data and validate it on the validation data
    model.fit(train_padded, train_labels,
              epochs=10,  # Train for 10 epochs
              validation_data=(val_padded, val_labels),
              verbose=1)  # Display training progress

    return model

# Main execution block: Train and save the model if the script is run directly
if __name__ == '__main__':
    # Train the model and save it to a file named "model.h5"
    model = solution_model()
    model.save("model.h5")
