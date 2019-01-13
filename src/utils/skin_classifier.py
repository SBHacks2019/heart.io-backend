from io import BytesIO
from PIL import Image

import numpy as np
from keras.models import model_from_json

import googleapiclient.discovery

import base64
import time
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-project-creds.json'

def load_model(modelpath, weightspath):
    with open(modelpath, "r") as file:
        loaded_json = file.read()
    skinmodel = model_from_json(loaded_json)

    # Load weights from file
    skinmodel.load_weights(weightspath)

    return skinmodel

def load_img(img_content, meanstdpath):
    # Load mean and std
    train_X_mean, train_X_std = np.load(meanstdpath)

    # Loading, resizing image as np.array
    imagearray = np.asarray(Image.open(BytesIO(img_content)).resize((100, 75)))
    imagearray = ((imagearray - train_X_mean) / train_X_std)

    return imagearray

def img_processor(img_content, meanstdpath, modelpath, weightspath):
    imagearray = load_img(img_content, meanstdpath)
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

def img_processor_online(img_content, meanstdpath, project, model, version=None):
    imagearray = load_img(img_content, meanstdpath)
    service = googleapiclient.discovery.build('ml', 'v1')
    name = f'projects/{project}/models/{model}'

    if version is not None:
        name += f'/versions/{version}'

    start_time = time.time()
    response = service.projects().predict(name=name, body={'instances': { 'input_image_bytes': imagearray.tolist() }}).execute()
    end_time = time.time()

    print(f'Inference took {end_time - start_time} seconds.')

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