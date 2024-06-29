from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2 as cv
import os

app = Flask(__name__)

# Recreate the model architecture
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(9, activation='softmax')  # Adjust the number of classes if needed
])

# Build the model by passing a sample input shape
model.build((None, 128, 128, 3))

# Load the model weights
model.load_weights('model.weights.h5')
print("Model weights loaded")

# Load class names
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Function to preprocess the image
def preprocess_image(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # converting from BGR colour scheme to RGB colour scheme
    img = cv.resize(img, (128, 128))  # Resize to 128x128 pixels
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')
        file = request.files['file']
        # if user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')
        if file:
            filepath = os.path.join('uploads', 'uploaded_image.jpg')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            img = preprocess_image(filepath)
            prediction = model.predict(img)
            index = np.argmax(prediction)
            result = class_names[index]
            return render_template('index.html', prediction=result)
    return render_template('index.html')

@app.route('/about-us.html')
def about_us():
    return render_template('about-us.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')


@app.route('/diagnosis-result.html')
def diagnosis_result():
    return render_template('diagnosis-result.html')


@app.route('/faq.html')
def faq():
    return render_template('faq.html')


@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/signup.html')
def signup():
    return render_template('signup.html')


@app.route('/symptoms.html')
def symptoms():
    return render_template('symptoms.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
