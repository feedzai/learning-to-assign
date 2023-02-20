import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import arange
from torch import float32 as torch_float32
from torch import isnan as torch_isnan
from torch import nn, no_grad, optim, tensor
from torch.utils import data as torch_data
from tqdm import tqdm


def only_y_true_squared_error(y_pred, y_true):
    criterion = nn.MSELoss(reduction='mean')
    w = (~torch_isnan(y_true))
    y_pred = y_pred[w]
    y_true = y_true[w]
    loss = criterion(y_pred, y_true)
    return loss

OPTIMIZERS = {
    'adam': optim.Adam,
    'sgd': optim.SGD
}
CRITERIONS = {
    'y_true_se': only_y_true_squared_error
}

class TorchDatasetFromPandas(torch_data.Dataset):

    def __init__(self, df: pd.DataFrame, label: str, device):
        self.df = df
        self.label = label

        self.X = tensor(self.df.drop(columns=[label]).values, dtype=torch_float32).to(device)
        self.y = tensor(self.df[label].values, dtype=torch_float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TorchDatasetFromXAndY(torch_data.Dataset):

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, device):
        self.X = tensor(X.values, dtype=torch_float32).to(device)
        self.y = tensor(y.values, dtype=torch_float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

TorchDataLoader = torch_data.DataLoader

def plot_losses(train_loss, val_loss):
    epochs = np.arange(start=1, stop=len(train_loss)+1, step=1)
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, val_loss, label='val')
    plt.title('Loss')
    plt.show()

def train_batch(X, y, model, optimizer, criterion):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    return loss

def train(model, train_dataloader, validation_dataset, n_epochs: int, optimizer: str, optimizer_kwargs, patience):
    """
    Train a PyTorch model.
    """
    opt = OPTIMIZERS[optimizer](
        model.parameters(),
        **optimizer_kwargs)
    criterion = CRITERIONS['y_true_se']

    # history keepers
    train_batch_losses = list()
    train_losses = list()
    val_losses = list()

    # early stopping vars
    best_model = model
    best_val_loss = np.inf
    n_epochs_without_progress = 0

    epochs = arange(1, n_epochs + 1)
    for _ in tqdm(epochs):
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, opt, criterion)
            train_batch_losses.append(loss.item())

        epoch_train_loss = tensor(train_batch_losses).mean().item()
        train_losses.append(epoch_train_loss)

        with no_grad():
            val_pred = model(validation_dataset.X)
            epoch_val_loss = criterion(y_pred=val_pred, y_true=validation_dataset.y)
            val_losses.append(epoch_val_loss.item())

        # early stopping
        if best_val_loss > epoch_val_loss:
            n_epochs_without_progress = 0
            best_val_loss = epoch_val_loss
            best_model = copy.deepcopy(model)
        else:
            n_epochs_without_progress += 1

        if n_epochs_without_progress >= patience:
            break

    plot_losses(train_losses, val_losses)

    return best_model

class DeepContextualBandit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepContextualBandit, self).__init__()

        self.i = input_dim
        self.o = output_dim

        self.h1 = nn.Linear(self.i, 2 * self.i)
        self.h2 = nn.Linear(2 * self.i, 4 * self.i)
        self.h3 = nn.Linear(4 * self.i, self.o)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.h1(x)
        x = self.ReLU(x)
        x = self.h2(x)
        x = self.ReLU(x)
        x = self.h3(x)

        return x
