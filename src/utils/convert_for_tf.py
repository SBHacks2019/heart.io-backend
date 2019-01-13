from keras.models import model_from_json
from keras import backend as K
import keras
import tensorflow as tf

import os
from shutil import rmtree

def convert_for_tf(modelpath, weightspath, export_path, clear_converted=False):
    K.set_learning_phase(0)

    model = None
    with open(modelpath, "r") as file:
        loaded_json = file.read()
        model = model_from_json(loaded_json)

    model.load_weights(weightspath)

    if clear_converted and os.path.exists(export_path):
        rmtree(export_path)

    with K.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={ 'input_image_bytes': model.input },
            outputs={ t.name: t for t in model.outputs }
        )

if __name__ == "__main__":
    print('Converting Keras model for use with Tensorflow...')
    convert_for_tf(
        modelpath='../ml-data/keras-files/skin-model.json',
        weightspath='../ml-data/keras-files/skin-model.h5',
        export_path='../ml-data/tf_export',
        clear_converted=True
    )
    print('Done!')