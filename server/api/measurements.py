from flask import request
from flask_restful import Resource
import torch
import numpy as np
import time

from extensions import classifier, preds_cache
from model import predict
import logging

class MeasurementsApi(Resource):
    def get(self):
        return preds_cache.load_item()

    def post(self):
        measurement = request.get_json()
        start_time = time.time()

        acc_x = -np.array(measurement['acc_y'])
        acc_y = -np.array(measurement['acc_x'])
        acc_z = -np.array(measurement['acc_z'])
        gyro_x = -np.array(measurement['gyro_y'])
        gyro_y = -np.array(measurement['gyro_x'])
        gyro_z = -np.array(measurement['gyro_z'])


        X = torch.tensor(np.array([gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]), dtype=torch.float).transpose(0, 1)

        pred, prob = predict(classifier, X)
        pred_time = time.time() - start_time

        result = {'pred': pred, 'prob': prob, 'pred_time': pred_time}
        preds_cache.save_item(result)

        return result