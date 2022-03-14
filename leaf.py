import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input
#from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
import tensorflow as tf
import numpy as np

#from tensorflow import keras
from tensorflow.keras import layers

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'leaf_deceased_detecion.hdf5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary

print('Model loading...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#graph = tf.get_default_graph()

print('Model loaded. Started serving...')

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(image_path):
    model = tf.keras.models.load_model('leaf_deceased_detecion.hdf5')
    img = load_img(image_path, target_size=(256, 256))
    image = img_to_array(img)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    class_names = ['Deceased_Alternaria Alternata','Deceased_Anthracnose','Deceased_Bacterial Blight','Deceased_Cercospora Leaf Spot','Healthy_Leaves']
    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])
    preds = [(class_names[np.argmax(score)]), (100 * np.max(score))]
    #print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(label[0],label[1]))
    return preds[0]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print('Begin Model Prediction...')

        # Make prediction
        preds = model_predict(file_path)

        print('End Model Prediction...')

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        
        return preds
    return None

if __name__ == '__main__':    
    app.run(debug=False, threaded=False)

    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
