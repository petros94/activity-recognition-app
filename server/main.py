from flask import Flask
from flask_cors import CORS

from api.finetuning import FineTuningSubmit, FineTuningStart
from api.measurements import MeasurementsApi
import yaml

from extensions import api

def create_app():
    print("Initializing app")
    with open("resources/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    app = Flask(__name__)
    CORS(app)
    api.add_resource(MeasurementsApi, '/measurements')
    api.add_resource(FineTuningSubmit, '/tuning/submit')
    api.add_resource(FineTuningStart, '/tuning/start')
    api.init_app(app)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000, host='0.0.0.0')
