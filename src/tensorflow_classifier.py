import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Bio import SeqIO
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras

# Preprocessing
data_directory = './data/common'  # Replace with your actual data directory
lineages = [i.replace('.fasta','') for i in os.listdir(data_directory) if i.endswith('.fasta')]

sequences = []
labels = []

# Read DNA sequences and assign labels
for lineage in lineages:
    file_path = f'{data_directory}/{lineage}.fasta'
    for record in SeqIO.parse(file_path, 'fasta'):
        sequences.append(record.seq)
        labels.append(lineages.index(lineage))

# Convert sequences to numerical representation (one-hot encoding)
nucleotides = ['A', 'T', 'C', 'G']
encoded_sequences = []
for seq in sequences:
    encoded_seq = []
    for nucleotide in seq:
        one_hot = [int(nucleotide == nuc) for nuc in nucleotides]
        encoded_seq.append(one_hot)
    encoded_sequences.append(encoded_seq)

# Pad sequences to ensure they have the same length
padded_sequences = pad_sequences(encoded_sequences)

# Convert to numpy arrays
x = np.array(padded_sequences)
y = np.array(labels)


# Now data is preprocessed data (x: input features, y: labels) ready for further steps.

# Split data into training and test sets
shuffled_indices = np.random.permutation(len(x))
shuffled_x = x[shuffled_indices]
shuffled_y = y[shuffled_indices]

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Split the data
x_train, x_val_test, y_train, y_val_test = train_test_split(
    shuffled_x, shuffled_y, test_size=(val_ratio + test_ratio), random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_val_test, y_val_test, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
)

# Verify the sizes of each set
print("Training set size:", len(x_train))
print("Validation set size:", len(x_val))
print("Testing set size:", len(x_test))

# Define the LSTM-based model
model = keras.Sequential()
model.add(keras.layers.LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(len(lineages), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Plot the learning curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Make predictions on the testing set
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)