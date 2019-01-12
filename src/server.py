import os.path

import flask
from flask import Flask, jsonify, request

from flask_cors import CORS

# from werkzeug.contrib.cache import FileSystemCache
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/input_files'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg' ])
MODEL_FILEPATH = 'model/model.pkl'

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from sklearn.externals import joblib

model = None
if os.path.exists(MODEL_FILEPATH):
    model = joblib.load(MODEL_FILEPATH)

allowed_file = lambda filename: '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

throw_error = lambda msg: jsonify({ 'error': msg })
res_success = lambda msg: jsonify({ 'success': msg })

@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify({ 'message': 'hello world' })

# define a predict function as an endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return throw_error('No file detected!')

    file = request.files['file']

    if file.filename == '':
        return throw_error('No file selected!')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        if model is not None:
            # TODO: implement classification
            return res_success('We got the image!')
        else:
            return throw_error('The model file is not present!')

# start the flask app, allow remote connections
if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True
    )
