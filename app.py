from flask import Flask, jsonify, request, url_for, render_template
from werkzeug.utils import secure_filename
from modules.nn import recognizer
import os
import cv2
import numpy as np
from PIL import Image
import glob

app = Flask(__name__, static_folder='static', static_url_path='/static')

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello_world():
    return jsonify({
        'message': 'You are connected to FLASK server'
    })


@app.route('/send', methods=['POST'])
def debug_post():
    return jsonify({
        'message': 'received',
        'data': request.json,
    })


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image):
    """
    Processing the image
    :param image:
    :return:
    """
    recognizer.recognize(image, target=f'{UPLOAD_FOLDER}/labelled')
    np_image = np.array(Image.open(image))
    filename = secure_filename(image.filename)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, filename), cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB))


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({
                'message': 'no file uploaded'
            })
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({
                'message': 'No file uploaded. Perhaps you forgot to select the file'
            })
        if file and allowed_file(file.filename):
            process_image(file)
            filename = secure_filename(file.filename)
            return jsonify({
                'message': 'Successfully uploaded',
                'data': {
                    'filename': url_for('upload_file', filename=filename),
                },
            })


@app.route('/home')
def home():
    images = glob.glob(f"{app.config['UPLOAD_FOLDER']}/labelled/*.jpg") +\
             glob.glob(f"{app.config['UPLOAD_FOLDER']}/labelled/*.jpeg")
    images = [{"name": os.path.basename(i), "src": i} for i in images]
    return render_template('home.html', images=images)
