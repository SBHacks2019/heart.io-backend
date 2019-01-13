from PIL import Image
import numpy as np
from keras.models import model_from_json

import googleapiclient.discovery

import base64

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'src/google-project-creds.json'

DEFAULT_MODEL_FILE = os.environ.get('MODEL_PATH')
DEFAULT_WEIGHTS_FILE = os.environ.get('WEIGHTS_PATH')
DEFAULT_MEANSTD_PATH = os.environ.get('MEANSTD_PATH')

GCP_PROJECT_NAME = os.environ.get('GCP_PROJECT_NAME')
ML_ENGINE_MODEL_NAME = os.environ.get('ML_ENGINE_MODEL_NAME')
ML_ENGINE_MODEL_VERSION = os.environ.get('ML_ENGINE_MODEL_VERSION')

def load_model(modelpath, weightspath):
    with open(modelpath, "r") as file:
        loaded_json = file.read()
    skinmodel = model_from_json(loaded_json)

    # Load weights from file
    skinmodel.load_weights(weightspath)

    return skinmodel

def load_img(imgpath, meanstdpath):
    # Load mean and std
    train_X_mean, train_X_std = np.load(meanstdpath)

    # Loading, resizing image as np.array
    imagearray = np.asarray(Image.open(imgpath).resize((100, 75)))
    imagearray = ((imagearray - train_X_mean) / train_X_std)

    return imagearray

def img_processor(imgpath, meanstdpath, modelpath, weightspath):
    imagearray = load_img(imgpath, meanstdpath)
    skinmodel = load_model(modelpath, weightspath)
    pred_vec = skinmodel.predict(imagearray).flatten().tolist()

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

def img_processor_online(imgpath, meanstdpath, project, model, version=None):
    imagearray = load_img(imgpath, meanstdpath)
    service = googleapiclient.discovery.build('ml', 'v1')
    name = f'projects/{project}/models/{model}'

    if version is not None:
        name += f'/versions/{version}'

    response = service.projects().predict(name=name, body={'instances': { 'input_image_bytes': imagearray.tolist() }}).execute()
    pred_vec = response['predictions'][0]['dense_4/Softmax:0']

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

if __name__ == '__main__':
    res = img_processor_online(
        imgpath="./google_scraped_data/actinic_keratoses/11. cutaneous-horn.jpg",
        modelpath=DEFAULT_MODEL_FILE,
        project=GCP_PROJECT_NAME,
        model=ML_ENGINE_MODEL_NAME,
        version=ML_ENGINE_MODEL_VERSION
    )

    print(res)