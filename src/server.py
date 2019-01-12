import os.path
from glob import glob
import json
from hashlib import md5

import flask
from flask import Flask, jsonify, request
from flask_cors import CORS

from werkzeug.utils import secure_filename

from errors.InvalidUsage import InvalidUsage
from skin_img_processor import img_processor

UPLOAD_FOLDER = '../uploads'
MODEL_ARCHS_FOLDER = '../models/archs/*.json'
MODEL_WEIGHTS_FOLDER = '../models/weights/*.h5'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg' ])
DEFAULT_FILE_KEY = 'input'

app = Flask(__name__)
CORS(app)

# loaded_models = {}

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

# def predict_with_model(image_path, model_name):
#     model = loaded_models.get(model_name)
#     print(image_path)
#     if model is not None:
#         # TODO: implement classification
#         return res_success('We got the image!')
#     else:
#         # this shouldn't happen at all
#         raise InvalidUsage('The model file is not present!')

# Flask methods

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    print(error)
    status_code = 500
    success = False
    response = {
        'success': success,
        'error': {
            'type': 'UnexpectedException',
            'message': 'An unexpected error has occurred.'
        }
    }

    return jsonify(response), status_code

@app.route('/hello', methods=['GET'])
def hello_world():
    return res_success('hello world')

@app.route('/predict-skin-mock', methods=['POST'])
def predict_skin_mock():
    image_path = save_and_get_input_file(request.files)
    return jsonify({
        "labels": ['Diseases', 'a', 'b', 'c', 'd', 'e', 'f', 'g'],
        "results": ['Probabilities', 1, 2, 3, 4, 5, 6, 7]
    })

@app.route('/predict-skin', methods=['POST'])
def predict_skin():
    image_path = save_and_get_input_file(request.files)
    #results = predict_with_model(image_path, 'skin-model')
    prediction_results = img_processor(
        image_path,
        meanstdpath="../ml/models/skin-model_meanstd.npy",
        modelpath="../ml/archs/skin-model.json",
        weightspath="../ml/weights/skin-model.h5"
    )

    labels = list(prediction_results.keys())
    results = list(prediction_results.values())
    new_results = [round(val, 7) * 100 for val in results]

    print(prediction_results)

    return jsonify({
        "labels": ['Diseases', *labels],
        "results": ['Results', *new_results]
    })

if __name__ == "__main__":
    # load all models into memory
    # print('Loading models...')
    # for (model_arch_path, model_weight_path) in zip(glob(MODEL_ARCHS_FOLDER), glob(MODEL_WEIGHTS_FOLDER)):
    #     print(model_arch_path, model_weight_path)
    #     model_name, _ = get_path_parts(model_arch_path, True)
    #     with open(model_arch_path) as f:
    #         model_arch = open(model_arch_path).read() #json.loads(f)
    #         model = model_from_json(model_arch)
    #         model.load_weights(model_weight_path)
    #         loaded_models[model_name] = model

    # print(f'{len(loaded_models)} models loaded.')
    # print(list(loaded_models.keys()))

    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True,
        threaded=False
    )