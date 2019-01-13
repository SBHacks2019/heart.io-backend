import os.path
from glob import glob
import json
from hashlib import md5

from io import BytesIO

# from PIL import Image
# import numpy as np

import flask
from flask import Flask, jsonify, request
from flask_cors import CORS

from werkzeug.utils import secure_filename

from errors.InvalidUsage import InvalidUsage
from utils.skin_classifier import img_processor, img_processor_online

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg' ])
DEFAULT_FILE_KEY = 'input'

app = Flask(__name__)
CORS(app)

allowed_file = lambda filename: '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
get_md5 = lambda string: md5(string.encode('utf-8')).hexdigest()
res_success = lambda msg: jsonify({ 'success': msg })

# custom methods

def get_input_file_content(files_dict):
    if DEFAULT_FILE_KEY not in files_dict:
        raise InvalidUsage('No file detected!')

    file = files_dict[DEFAULT_FILE_KEY]

    if file.filename == '':
        raise InvalidUsage('No file selected!')

    return file.read()

# Flask methods

going_online = True

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/predict-skin', methods=['POST'])
def predict_skin():
    img_content = get_input_file_content(request.files)

    prediction_results = None
    if going_online:
        prediction_results = img_processor_online(
            img_content,
            meanstdpath='../ml-data/keras-files/skin-model_meanstd.npy',
            project='project-2b1s',
            model='skin_disease_detection',
            version='final'
        )
    else:
        prediction_results = img_processor(
            img_content,
            meanstdpath='../ml-data/keras-files/skin-model_meanstd.npy',
            modelpath='../ml-data/keras-files/skin-model.json',
            weightspath='../ml-data/keras-files/skin-model.h5'
        )

    labels = list(prediction_results.keys())
    results = list(prediction_results.values())
    new_results = [round(val, 6) * 100 for val in results]

    print(prediction_results)

    return jsonify({
        "labels": ['Diseases', *labels],
        "results": ['Results', *new_results]
    })

if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True,
        threaded=False,
        use_reloader=False
    )