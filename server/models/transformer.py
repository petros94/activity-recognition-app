from torch import nn
import torch
from torch.nn import functional as F

from constants import LABELS


class Transformer_classifier(nn.Module):
    def __init__(self, input_size, d_model, dim_feedforward, num_layers, output_size) -> None:
        super(Transformer_classifier, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        seq_size = x.size(dim=0)
        batch_size = x.size(dim=1)
        feature_size = x.size(dim=-1)

        # Padd dimensions
        pad = torch.tensor([0])
        pad = pad.expand(seq_size, batch_size, self.d_model - feature_size)
        x = torch.cat((x, pad), dim=-1)

        feature_size = x.size(dim=-1)

        cls = torch.tensor([1])
        cls = cls.expand(1, batch_size, feature_size)
        input = torch.cat((cls, x), dim=0)

        # x = self.embedding(input)
        x = input
        # x = self.pos_encoding(x)
        out = self.transformer(x)
        # cat = torch.cat((out[-1, :, :], out[-2, :, :]), dim=1)
        # cat = torch.mean(out, dim=0)
        cat = out[0, :, :]
        out = self.fc(cat)
        return out