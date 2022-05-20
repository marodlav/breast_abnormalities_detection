import base64
import mimetypes
import os
import platform
import subprocess
import time
import flask
import errors
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './image_uploads'
RESULT_FOLDER = './image_results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# APP SERVER
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


def allowed_file(filename):
    # Check for allowed file extension on uploaded image
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_filemame(_file):
    # Generate file name with {{timestamp.extension}} format
    ts_milliseconds = int(round(time.time() * 1000))
    file_extension = os.path.splitext(_file.filename)[1]
    file_name = secure_filename(str(ts_milliseconds)+file_extension)
    return file_name


def save_file(file):
    # Save file on UPLOADS folder and provide the paths for files
    file_name = generate_filemame(file)
    upload_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(upload_file_path)
    result_file_path = os.path.join(app.config['RESULT_FOLDER'], file_name)
    return upload_file_path, result_file_path


def delete_file(path):
    # Delete uploaded and resulted file
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        print(errors.ERROR_DELETING_FILE, e)


def encode_result_file(result_file_path):
    # Encode result image file to BASE64 so the files can be deleted
    # after response without aftecting user
    if (os.path.exists(result_file_path)):
        try:
            file_mimetype = mimetypes.guess_type(result_file_path)[0]
            result_file = open(result_file_path, 'rb')
            encoded_image = base64.b64encode(result_file.read())
            image_src = 'data:' + file_mimetype + \
                ';base64,' + encoded_image.decode('utf8')
            return image_src
        except Exception as e:
            print(errors.ERROR_FILE_ENCODING, e)
            return False
    else:
        print(errors.ERROR_FILE_NOT_FOUND)
        return False


def process_image(upload_file_path, result_file_path):
    # PLACE HERE THE CODE EXCUTION FOR YOLOV DETECTION
    # The result image need to be storage on the 'results' folder
    # In the meantime, we use a subprocess call to copy the uploaded image
    # to the results folder with the same name and extension

    if platform.system() == 'Windows':
        exit_code = subprocess.call(
            'copy ' + upload_file_path + ' ' + result_file_path, shell=True)
    else:
        exit_code = subprocess.call(
            'cp ' + upload_file_path + ' ' + result_file_path, shell=True)
    time.sleep(3)
    return exit_code


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in flask.request.files:
        return errors.ERROR_NOT_FILE_UPLOAD, 400
    file = flask.request.files['file']
    if file.filename == '':
        return errors.ERROR_NOT_FILE_UPLOAD, 400
    if not allowed_file(file.filename):
        return errors.ERROR_EXTENSION_NOT_ALLOWED, 400

    # Save uploaded image
    upload_file_path, result_file_path = save_file(file)

    # Setup post-exectute to try to delete both files just after sending response
    @flask.after_this_request
    def add_close_action(response):
        @response.call_on_close
        def process_after_request():
            delete_file(upload_file_path)
            delete_file(result_file_path)
        return response

    ################# YOLOV MAGIG ####################################

    exit_code = process_image(upload_file_path, result_file_path)
    if exit_code != 0:  # Exit with code ! = 0 means error on command exectuion
        print(errors.ERROR_PROCESSING)
        return errors.ERROR_PROCESSING, 400

    ##################################################################

    # Read and encode processed file for response
    response = encode_result_file(result_file_path)
    if response:
        return response
    else:
        return errors.ERROR_PROCESSING, 400
