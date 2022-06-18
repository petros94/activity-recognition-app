import torch
import yaml
from flask_restful import Api

from constants import LABELS
from model import LSTM
from repository import InMemoryStorage

with open("resources/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

classifier = LSTM(input_size=config['model']['input_size'],
                      hidden_size=config['model']['hidden_size'],
                      num_layers=config['model']['num_layers'],
                      output_size=config['model']['output_size'],
                      bidirectional=config['model']['bidirectional'])

api = Api()

preds_cache = InMemoryStorage(config['cache']['max_size'])
calibration_cache = {k: InMemoryStorage(config['cache']['max_size']) for k in LABELS}