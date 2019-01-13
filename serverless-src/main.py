import flask

import numpy as np
from io import BytesIO
from PIL import Image

import googleapiclient.discovery

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-project-creds.json'

DEFAULT_FILE_KEY = 'input'

# shamelessly copied from Flask docs
# source: http://flask.pocoo.org/docs/1.0/patterns/apierrors/
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['error'] = self.message
        return rv

def get_input_file_content(files_dict):
    if DEFAULT_FILE_KEY not in files_dict:
        raise InvalidUsage('No file detected!')

    file = files_dict[DEFAULT_FILE_KEY]

    if file.filename == '':
        raise InvalidUsage('No file selected!')

    return file.read()

def load_img(img_content, meanstdpath):
    # Load mean and std
    train_X_mean, train_X_std = np.load(meanstdpath)

    # Loading, resizing image as np.array
    imagearray = np.asarray(Image.open(BytesIO(img_content)).resize((100, 75)))
    imagearray = ((imagearray - train_X_mean) / train_X_std)

    return imagearray

def img_processor_online(img_content, meanstdpath, project, model, version=None):
    imagearray = load_img(img_content, meanstdpath)
    service = googleapiclient.discovery.build('ml', 'v1')
    name = f'projects/{project}/models/{model}'

    if version is not None:
        name += f'/versions/{version}'

    response = service.projects().predict(name=name, body={'instances': { 'input_image_bytes': imagearray.tolist() }}).execute()
    pred_vec = response['predictions'][0]['dense_2/Softmax:0']

    pred_dict = {
        'Actinic keratoses' : pred_vec[0],
        'Basal cell carcinoma' : pred_vec[1],
        'Benign keratosis-like lesions' : pred_vec[2],
        'Dermatofibroma' : pred_vec[3],
        'Melanocytic nevi' : pred_vec[4],
        'Melanoma' : pred_vec[5],
        'Vascular lesions' : pred_vec[6]
    }

    return pred_dict

def predict_skin(request):
    img_content = get_input_file_content(request.files)

    prediction_results = img_processor_online(
        img_content,
        meanstdpath='skin-model_meanstd.npy',
        project='project-2b1s',
        model='skin_disease_detection',
        version='final'
    )

    labels = list(prediction_results.keys())
    results = list(prediction_results.values())
    new_results = [round(val, 6) * 100 for val in results]

    print(prediction_results)

    response = flask.jsonify({
        "labels": ['Diseases', *labels],
        "results": ['Results', *new_results]
    })

    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST')

    return response