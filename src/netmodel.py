import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy

class Net(pl.LightningModule):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_shape[0], 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        # self.conv3 = nn.Conv1d(64,128,3,padding=1)
        self.fc1 = nn.Linear(input_shape[1] * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=65)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log('train_accuracy', accuracy)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log('val_accuracy', accuracy)
        self.log('val_loss', loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
