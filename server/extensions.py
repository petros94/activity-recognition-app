import torch
import yaml
from flask_restful import Api

from constants import LABELS

from models.factory import load_model
from repository import InMemoryStorage

with open("resources/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

classifier = load_model()

api = Api()

preds_cache = InMemoryStorage(config['cache']['max_size'])
calibration_cache = {k: InMemoryStorage(config['cache']['max_size']) for k in LABELS}