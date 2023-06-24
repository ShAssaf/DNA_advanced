import os


import torch
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
from Bio import SeqIO
from Bio.Seq import translate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sympy import evaluate
from torch import flatten
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.common_func import filter_nonstandard, load_data
from src.netmodel import Net


# Load data from fasta files


# One hot encoding
def one_hot_encoding(data):
    # remove 'N' from sequences and then one-hot encode
    data = {lineage: [filter_nonstandard(seq) for seq in sequences] for lineage, sequences in data.items()}
    encoder = LabelBinarizer()
    data_encoded = {lineage: [encoder.fit_transform(list(seq)) for seq in sequences]
                    for lineage, sequences in data.items()}
    return data_encoded, encoder.classes_

# Prepare data for the model
def prepare_data(data_encoded, classes):
    X = []
    y = []
    min_seq_len = min([len(arr) for values in data_encoded.values() for arr in values])
    for lineage, sequences in data_encoded.items():

        for sequence in sequences:
            X.append(sequence[:min_seq_len])
            y.append(lineage)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, le.classes_


# Define the model



# Load data
data_directory = './data/tmp'
print("loading data...")
data = load_data(data_directory)
print("encoding data")
data_encoded, classes = one_hot_encoding(data)
print("preparing data")
X_train, X_test, y_train, y_test, classes = prepare_data(data_encoded, classes)

# Convert data to torch tensors
print("converting data...")
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
print("creating loaders")
train_data = torch.utils.data.TensorDataset(X_train, y_train)
test_data = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=32)
val_loader = DataLoader(test_data, batch_size=32)
print("creating model")
# Create model
model = Net(input_shape=X_train.shape[1:], num_classes=len(classes))

# Train the model
print("training model")
trainer = pl.Trainer(max_epochs=100, devices=1) #, accelerator="gpu"
trainer.fit(model, train_loader, val_loader)
# Retrieve the training and validation losses from the trainer object


# Set model to evaluation mode
model.eval()

# Initialize variables for accuracy calculation
correct = 0
total = 0

# Iterate over test dataloader and compute predictions
with torch.no_grad():
    for inputs, labels in val_loader:

        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Update accuracy count
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = correct / total
print('Test Accuracy:', accuracy)
