# Importing necessary modules
import numpy as np
import os
import glob
import re
import sys

# keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Flask utils
from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
app = Flask(__name__)
model_path = 'vgg16.h5'

## Load model
model = load_model(model_path)
model.make_predict_function()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    # preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  ## Using html we can upload image there(Home page with Choose Button)


@app.route('/predict', methods=['GET', 'POST'])  ## Predict button
def upload():
    if request.method == 'POST':
        # Get the file from Post
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(filepath)

        ## Here we make prediction
        pred = model_predict(filepath, model)  # class index
        pred_class = decode_predictions(pred, top=1)  # map class index to class label
        result = str(pred_class[0][0][1])
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
