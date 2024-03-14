import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load the dataset
train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_df = pd.read_csv(train_file_path, sep='\t', names=["label", "message"])
test_df = pd.read_csv(test_file_path, sep='\t', names=["label", "message"])

# Preprocess labels
train_df['label'] = train_df['label'].map({'ham': 0, 'spam': 1})
test_df['label'] = test_df['label'].map({'ham': 0, 'spam': 1})

# Prepare datasets
train_labels = train_df.pop('label')
test_labels = test_df.pop('label')

# Define the TextVectorization layer
max_features = 1000
max_length = 1000
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=max_length)

# Fit the TextVectorization layer to the training text
vectorize_layer.adapt(train_df['message'].values)

# Vectorize the messages
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = tf.data.Dataset.from_tensor_slices((train_df['message'].values, train_labels.values)).batch(32).map(vectorize_text)
test_ds = tf.data.Dataset.from_tensor_slices((test_df['message'].values, test_labels.values)).batch(32).map(vectorize_text)

# Define the model with Bidirectional LSTM
model = Sequential([
    Embedding(max_features + 1, 64, mask_zero=True),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Setup EarlyStopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max', restore_best_weights=True)

# Train the model
model.fit(train_ds, epochs=10, validation_data=test_ds, callbacks=[early_stop], verbose=1)

# Predict message function remains the same, just adjust how text is prepared
def predict_message(pred_text):
    pred_text = vectorize_layer(tf.expand_dims(pred_text, -1))
    prediction = model.predict(pred_text)[0][0]
    label = "spam" if prediction >= 0.5 else "ham"
    # print(f"Message: {pred_text}\nPredicted: {label} ({prediction:.4f})")
    return [prediction, label]

# pred_text = "how are you doing today?"

# prediction = predict_message(pred_text)
# print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      print(f"Failed on: {msg}, predicted: {prediction}, expected: {ans}")
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()