import os
import re

import pytorch_lightning as pl
import torch
import torch.nn as nn
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader


# Load data from fasta files
def load_data(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".fasta"):
            lineage = filename.replace(".fasta", "")
            sequences = list(SeqIO.parse(os.path.join(directory, filename), "fasta"))
            data[lineage] = [str(seq.seq) for seq in sequences]
    return data


def filter_nonstandard(sequence):
    filtered_sequence = re.sub(r'[^ACGT]', '', sequence)
    return filtered_sequence


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
class Net(pl.LightningModule):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_shape[0], 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(input_shape[1] * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Load data
data_directory = './data/common'
data = load_data(data_directory)
data_encoded, classes = one_hot_encoding(data)

X_train, X_test, y_train, y_test, classes = prepare_data(data_encoded, classes)

# Convert data to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
train_data = torch.utils.data.TensorDataset(X_train, y_train)
test_data = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=32)
val_loader = DataLoader(test_data, batch_size=32)

# Create model
model = Net(input_shape=X_train.shape[1:], num_classes=len(classes))

# Train the model
trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1)
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
