import yaml

from constants import LABELS
from models.lstm import LSTM
from models.transformer import Transformer_classifier
from torch.nn import functional as F
import torch


def predict(model, x):
    x = x.unsqueeze(1)
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(x).squeeze(1)
        return LABELS[F.softmax(y_pred).argmax(-1).numpy()[0]], F.softmax(y_pred).max().item()


def load_model():
    with open("resources/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args = config['model']['args']
    type = config['model']['type']
    if type == 'transformer':
        classifier = Transformer_classifier(**args)
    elif type == 'lstm':
        classifier = LSTM(**args)
    else:
        raise ValueError("Invalid model type")

    classifier.load_state_dict(torch.load(config['model']['path'], map_location="cpu"))
    classifier.cpu()
    classifier.eval()
    return classifier
