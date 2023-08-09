import numpy
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from pytorch_lightning import loggers

from src.common_func import load_data
from src.netmodel import Net


def Kmers_funct(seq, size):
    return [seq[x:x + size].lower() for x in range(len(seq) - size + 1)]


def data_into_kmers_count(data):
    cv = CountVectorizer()
    k_mers_count_dict = {}
    for lineage, sequences in data.items():
        k_mers_sequences = []
        k_mers_sequences.append(generate_all_combs())
        for sequence in sequences:
            k_mers_sequences.append(' '.join(Kmers_funct(sequence, size=4)))
        k_mers_sequences_count = cv.fit_transform(k_mers_sequences).toarray()

        k_mers_count_dict[lineage] = numpy.array(k_mers_sequences_count)
    return k_mers_count_dict


def generate_all_combs():
    from itertools import product
    li = ['A', 'T', 'G', 'C']
    combinations = []
    for comb in product(li, repeat=len(li)):
        combinations.append(''.join(comb))
    return ' '.join(combinations)


def k_merprepare_data(data_encoded):
    X = []
    y = []
    size = 100
    min_seq_len = min([len(arr) for values in data_encoded.values() for arr in values])
    resize_len = size - min_seq_len % size
    for lineage, sequences in data_encoded.items():

        for sequence in sequences:
            X.append(np.resize(sequence[:min_seq_len], min_seq_len + resize_len).reshape(-1, size))
            y.append(lineage)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, le.classes_


data_directory = './data/common'
data = load_data(data_directory)
# load kmer_count_dict
k_mers_count_dict = {}
for lineage in data.keys():
    k_mers_count_dict[lineage] = np.load(f'./data/kmers_dict/{lineage}.npy')



X_train, X_test, y_train, y_test, classes = k_merprepare_data(k_mers_count_dict)

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
print("input_shape", X_train.shape[1:], " num_classes", len(classes))
# Train the model

tb_logger = loggers.TensorBoardLogger('logs/')
trainer = pl.Trainer(max_epochs=25, logger=tb_logger, accelerator="gpu")
trainer.fit(model, train_loader, val_loader)
# Retrieve the training and validation losses from the trainer object


# save model
torch.save(model.state_dict(), './model/kmer_model.pth')
# Set model to evaluation mode
model.eval()



# Initialize variables for accuracy calculation
# Initialize variables for accuracy and loss calculation
correct = 0
total = 0
loss_function = torch.nn.CrossEntropyLoss()  # Use the same loss function as during training
losses = []

# Iterate over test dataloader and compute predictions
with torch.no_grad():
    for inputs, labels in val_loader:
        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Calculate loss
        loss = loss_function(outputs, labels)
        losses.append(loss.item())

        # Update accuracy count
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = correct / total
print('Test Accuracy:', accuracy)

# Calculate accuracy
accuracy = correct / total
print('Test Accuracy:', accuracy)
