import os.path
from glob import glob

import flask
from flask import Flask, jsonify, request
from flask_cors import CORS

# from werkzeug.contrib.cache import FileSystemCache
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '../uploads'
MODELS_FOLDER = '../models/*.pkl'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg' ])
DEFAULT_FILE_KEY = 'input'

app = Flask(__name__)
CORS(app)

from sklearn.externals import joblib

from errors.InvalidUsage import InvalidUsage

loaded_models = {}

allowed_file = lambda filename: '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

res_success = lambda msg: jsonify({ 'success': msg })

# custom methods

def save_and_get_input_file(files_dict):
    if DEFAULT_FILE_KEY not in files_dict:
        raise InvalidUsage('No file detected!')

    file = request.files[DEFAULT_FILE_KEY]

    if file.filename == '':
        raise InvalidUsage('No file selected!')
    elif file and allowed_file(file.filename):
        # construct the file name and path
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        filedir = os.path.dirname(filepath)

        # create the necessary directories (if applicable) and save the image
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        file.save(filepath)
        return filepath
    else:
        raise InvalidUsage('Invalid file provided!')

def predict_with_model(image_path, model_name):
    model = loaded_models.get(model_name)
    print(image_path)
    if model is not None:
        # TODO: implement classification
        return res_success('We got the image!')
    else:
        # this shouldn't happen at all
        raise InvalidUsage('The model file is not present!')

# Flask methods

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/hello', methods=['GET'])
def hello_world():
    return res_success('hello world')

@app.route('/predict-mole', methods=['POST'])
def predict_mole():
    image_path = save_and_get_input_file(request.files)
    results = predict_with_model(image_path, 'mole-model')
    return results

@app.route('/predict-chest', methods=['POST'])
def predict_chest():
    image_path = save_and_get_input_file(request.files)
    results = predict_with_model(image_path, 'chest-model')
    return results

if __name__ == "__main__":
    # load all models into memory
    print('Loading models...')
    for model_path in glob(MODELS_FOLDER):
        print(model_path)
        model_name = os.path.basename(model_path)
        model_name_pure, model_name_ext = os.path.splitext(model_name)
        loaded_models[model_name_pure] = joblib.load(model_path)
    print(f'{len(loaded_models)} models loaded.')
    print(loaded_models)

    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True
    )