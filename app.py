from flask import Flask, jsonify, request, url_for, render_template
import os
import glob
import logging
from modules.nn import recognizer


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

app = Flask(__name__, static_folder='static', static_url_path='/static')

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello_world():
    return jsonify({'message': 'You are connected to FLASK server'})


@app.route('/send', methods=['POST'])
def debug_post():
    return jsonify({'mesage': 'received', 'data': request.json})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image, meta):
    return recognizer.recognize(
        image,
        target=f'{UPLOAD_FOLDER}/labelled',
        debug=f'{UPLOAD_FOLDER}/debug',
        meta=meta,
    )


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'message': 'no file uploaded'})
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify(
                {'message': 'No file uploaded. Perhaps you forgot to select the file'}
            )
        if file and allowed_file(file.filename):
            # sensor value
            meta = request.form.to_dict()
            filename = file.filename
            logging.info(f'Processing image {filename}')
            detections = process_image(file, meta=meta)
            logging.info(f'Processing image {filename} finished')
            return jsonify(
                {
                    'message': 'Successfully uploaded',
                    'data': {
                        'filename': url_for('upload_file', filename=filename),
                        'detections': detections,
                    },
                }
            )


@app.route('/home')
def home():
    images = glob.glob(f"{app.config['UPLOAD_FOLDER']}/labelled/*.jpg") + glob.glob(
        f"{app.config['UPLOAD_FOLDER']}/labelled/*.jpeg"
    )
    images = [{"name": os.path.basename(i), "src": i} for i in images]
    images = sorted(images, key=lambda x: x['name'], reverse=True)
    return render_template('home.html', images=images)
