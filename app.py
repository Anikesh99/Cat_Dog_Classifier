import flask
from flask import request as request

import urllib.request

from werkzeug.utils import secure_filename

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

import pickle

import os


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


with open(f'model/finalizedModel.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')
upload_folder = '/uploads/'
app.config['UPLOAD_FOLDER'] = '/uploads/'
class_names = ['Cat', 'Dog']

test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


@app.route('/')
def index():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')


@app.route('/', methods=['POST'])
def submit_file():
    if flask.request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            flask.flash('No file selected for uploading')
            return flask.redirect(request.url)
        if uploaded_file:
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))
            getPrediction(upload_folder+uploaded_file.filename)
            # acc = getPrediction(upload_folder+uploaded_file.filename)
            # flask.flash(acc)
            return flask.redirect('/')


def getPrediction(filename):
    image = Image.open(filename)
    im = test_transform(image)
    prediction = model(im.view(1, 3, 224, 224)).argmax()
    print(class_names[prediction.item()])
    flask.flash(class_names[prediction.item()])


if __name__ == '__main__':
    app.secret_key = 'whodis'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run()