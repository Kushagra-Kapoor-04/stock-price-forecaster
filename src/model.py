import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Stacked LSTM for stock price prediction (many-to-one).

    Architecture:
      LSTM(input=1, hidden=64, layers=2, dropout=0.2)
        → Dropout(0.2)
        → Linear(64 → 1)

    Why take only the last timestep?
      We're doing many-to-one prediction: 30 days in → 1 price out.
      out[:, -1, :] is the hidden state after processing all timesteps —
      it summarises the entire input sequence.

    Why 2 layers?
      Layer 1 captures short-term price patterns.
      Layer 2 captures longer-term trends built on layer 1's output.
      More than 2–3 layers rarely helps for financial data.
    """

    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # input shape: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        # Take only the last timestep's output
        out = self.dropout(out[:, -1, :])   # (batch, hidden_size)
        return self.fc(out)                 # (batch, 1)
