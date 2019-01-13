from keras.models import model_from_json
from keras import backend as K
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

    init = tf.global_variables_initializer()

    with tf.keras.backend.get_session() as sess:
        sess.run(init)
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={ 'input_image_bytes': model.input },
            outputs={ t.name: t for t in model.outputs }
        )