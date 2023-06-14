

//sentiment analysis using lstm code you can download dataset from kaggle

import numpy as np
import pandas as pd
import emoji
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# Load the data
mapping = pd.read_csv("D:\Mapping.csv")
output = pd.read_csv("D:\OutputFormat.csv")
train = pd.read_csv("D:\Train.csv")
test = pd.read_csv("D:\Test.csv")

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['TEXT'])
vocab_size = len(tokenizer.word_index) + 1

train_sequences = tokenizer.texts_to_sequences(train['TEXT'])
test_sequences = tokenizer.texts_to_sequences(test['TEXT'])

max_seq_length = 100  # define the maximum sequence length
train_data = pad_sequences(train_sequences, maxlen=max_seq_length)
test_data = pad_sequences(test_sequences, maxlen=max_seq_length)

# Check unique values in the label column
unique_labels = train['Label'].unique()
num_classes = len(unique_labels)


# Preprocess labels
label_mapping = {label: index for index, label in enumerate(unique_labels)}
train_labels = train['Label'].map(label_mapping)
train_labels = to_categorical(train_labels, num_classes=num_classes)


# Define the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_seq_length))
model.add(LSTM(128))
model.add(Dense(num_classes, activation='softmax'))


# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=5, batch_size=32)

# Make predictions on the testing data
test_probabilities = model.predict(test_data)
test_predictions = np.argmax(test_probabilities, axis=1)


print(mapping.columns)
print(mapping.head())


predicted_emojis = mapping['emoticons'][test_predictions]



# Loop through each sentence in the test data
for sentence in test['TEXT']:
    # Preprocess the sentence
    sequence = tokenizer.texts_to_sequences([sentence])
    data = pad_sequences(sequence, maxlen=max_seq_length)
    
    # Make prediction
    prediction = model.predict(data)
    predicted_label = np.argmax(prediction)
    
    # Map the predicted label to the emoji
    predicted_emoji = mapping['emoticons'][predicted_label]
    
    # Print the predicted emoji for the sentence
    print(f"Sentence: {sentence}")
    print(f"Predicted Emoji: {predicted_emoji}")
    print()










