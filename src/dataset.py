import torch
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    """
    PyTorch Dataset wrapping sliding-window sequences.

    Reshapes X from (batch, seq_len) → (batch, seq_len, 1)
    because LSTM expects input shape: (batch, seq_len, features).
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    """
    Build train and test DataLoaders.

    shuffle=False for both — temporal order must be preserved
    in time series (even for training, shuffling can hurt LSTM
    performance when sequences overlap heavily).
    """
    train_loader = DataLoader(
        StockDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        StockDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader
