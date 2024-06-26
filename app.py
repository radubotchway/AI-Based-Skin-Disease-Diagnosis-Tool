from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2 as cv
import os

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('skin_disease_classifier.keras')

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

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)