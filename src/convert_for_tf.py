from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf

import os
from shutil import rmtree

from dotenv import load_dotenv
load_dotenv()

DEFAULT_MODEL_FILE = os.environ.get('MODEL_PATH')
DEFAULT_WEIGHTS_FILE = os.environ.get('WEIGHTS_PATH')
DEFAULT_TF_EXPORT_PATH = os.environ.get('TF_MODEL_EXPORT_PATH')

def convert_for_tf(modelpath, weightspath, export_path, clear_converted=False):
    K.set_learning_phase(0)

    model = None
    with open(modelpath, "r") as file:
        loaded_json = file.read()
        model = model_from_json(loaded_json)

    model.load_weights(weightspath)

    if clear_converted and os.path.exists(export_path):
        rmtree(export_path)

    init = tf.global_variables_initializer()

    with tf.keras.backend.get_session() as sess:
        sess.run(init)
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={ 'input_image_bytes': model.input },
            outputs={ t.name: t for t in model.outputs }
        )

if __name__ == "__main__":
    print('Converting Keras model for use with Tensorflow...')
    convert_for_tf(
        modelpath=DEFAULT_MODEL_FILE,
        weightspath=DEFAULT_WEIGHTS_FILE,
        export_path=DEFAULT_TF_EXPORT_PATH,
        clear_converted=True
    )
    print('Done!')