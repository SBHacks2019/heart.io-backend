import os.path
from glob import glob
import json
from hashlib import md5

import flask
from flask import Flask, jsonify, request
from flask_cors import CORS

from werkzeug.utils import secure_filename

from errors.InvalidUsage import InvalidUsage
from utils.skin_classifier import img_processor_online

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg' ])
DEFAULT_FILE_KEY = 'input'

app = Flask(__name__)
CORS(app)

allowed_file = lambda filename: '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
get_md5 = lambda string: md5(string.encode('utf-8')).hexdigest()
res_success = lambda msg: jsonify({ 'success': msg })

# custom methods

def get_path_parts(file_name_or_path, is_path = False):
    base_name = os.path.basename(file_name_or_path) if is_path else file_name_or_path
    pure_name, ext = os.path.splitext(base_name)
    return pure_name, ext

def save_and_get_input_file(files_dict):
    if DEFAULT_FILE_KEY not in files_dict:
        raise InvalidUsage('No file detected!')

    file = request.files[DEFAULT_FILE_KEY]

    if file.filename == '':
        raise InvalidUsage('No file selected!')
    elif file and allowed_file(file.filename):
        # construct the file name and path
        orig_filename = secure_filename(file.filename)
        part_name, part_ext = get_path_parts(orig_filename)
        filename = get_md5(part_name) + part_ext
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        if not os.path.isfile(filepath):
            # create the necessary directories (if applicable) and save the image
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            file.save(filepath)

        return filepath
    else:
        raise InvalidUsage('Invalid file provided!')

# Flask methods

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/predict-skin', methods=['POST'])
def predict_skin():
    image_path = save_and_get_input_file(request.files)
    prediction_results = img_processor_online(
        image_path,
        meanstdpath='../ml-data/keras-files/skin-model_meanstd.npy',
        project='project-2b1s',
        model='skin_disease_detection',
        version='demo'
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