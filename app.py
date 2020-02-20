from flask import Flask, jsonify, request, url_for, render_template
from werkzeug.utils import secure_filename
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
    np_image = np.array(Image.open(image))
    rotated = cv2.rotate(np_image, cv2.ROTATE_90_CLOCKWISE)
    return cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)


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
                'message': 'No file uploaded'
            })
        if file and allowed_file(file.filename):
            processed_image = process_image(file)
            filename = secure_filename(file.filename)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), processed_image)
            return jsonify({
                'message': 'Successfully uploaded',
                'data': {
                    'filename': url_for('upload_file', filename=filename),
                },
            })


@app.route('/home')
def home():
    images = glob.glob(f"{app.config['UPLOAD_FOLDER']}/*.jpg")
    return render_template('home.html', images=images)
