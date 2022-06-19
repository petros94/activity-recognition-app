import time

import torch
import yaml
from flask import request
from flask_restful import Resource
import numpy as np
from sklearn.model_selection import train_test_split

from constants import LABELS
from extensions import calibration_cache, classifier, preds_cache
from finetuning import HARDataset, train_model
from models.factory import predict

with open("resources/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class FineTuningSubmit(Resource):
    def delete(self):
        for idx, type in enumerate(LABELS):
            calibration_cache[type].data = []
        return {'status': 'ok'}

    def post(self):
        start_time = time.time()
        measurement = request.get_json()
        type = measurement['type']

        acc_x = -np.array(measurement['acc_y'])
        acc_y = -np.array(measurement['acc_x'])
        acc_z = -np.array(measurement['acc_z'])
        gyro_x = -np.array(measurement['gyro_y'])
        gyro_y = -np.array(measurement['gyro_x'])
        gyro_z = -np.array(measurement['gyro_z'])



        mes = np.array([gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z])
        mes = np.transpose(mes)
        calibration_cache[type].save_item(list(mes))

        X = torch.tensor(mes, dtype=torch.float)

        pred, prob = predict(classifier, X)
        pred_time = time.time() - start_time

        result = {'pred': pred, 'prob': prob, 'pred_time': pred_time}
        preds_cache.save_item(result)

        return result

class FineTuningStart(Resource):
    def delete(self):
        classifier.load_state_dict(torch.load(config['model']['path'], map_location="cpu"))
        classifier.cpu()
        classifier.eval()
        return {'status': 'ok'}

    def post(self):

        X_train = []
        y_train = []
        for idx, type in enumerate(LABELS):
            samples = np.array(calibration_cache[type].data)
            if len(samples) > 0:
                X_train.extend(samples)
                y_train.extend(np.ones(len(samples))*idx)

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

        train_dataset = HARDataset(X_train, y_train)
        test_dataset = HARDataset(X_test, y_test)

        args = {
            'max_epochs': 2,
            'batch_size': 16,
            'save_path': 'resources/temp_model.pt'
        }

        stats = train_model(classifier, train_dataset, test_dataset, **args)
        classifier.load_state_dict(torch.load('resources/temp_model.pt', map_location="cpu"))
        classifier.cpu()
        classifier.eval()

        print("Training complete")
        return stats


