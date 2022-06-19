from torch import nn
from torch.nn import functional as F
import torch

from constants import LABELS


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional) -> None:
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=bidirectional, dropout=0.2)
        self.output_size = output_size
        self.fc = nn.Linear(in_features=(2 if bidirectional else 1) * self.hidden_size, out_features=output_size)

    def forward(self, x):
        out_packed, (h, c) = self.lstm(x)
        cat = torch.cat((h[-1, :, :], h[-2, :, :]), dim=1) if self.bidirectional else h[-1, :, :]
        x = self.fc(cat)
        return x.view(-1, self.output_size)
