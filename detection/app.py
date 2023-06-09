from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
import sys
sys.path.append('./utils')
app = Flask(__name__)


uploads_dir = os.path.join(app.instance_path, 'uploads')

os.makedirs(uploads_dir, exist_ok=True)
@app.route('/predict', methods=['GET','POST'])
def detect():
    subprocess.run("ls")
    subprocess.run(['python3', 'detect.py', '--source', 'data1.mp4'])

    # return os.path.join(uploads_dir, secure_filename(video.filename))
    #obj = secure_filename(video.filename)
    #return obj


if __name__ == "__main__":
    detect()
    app.run()