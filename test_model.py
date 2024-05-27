import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('skin_disease_classifier.keras')

# Load class names
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Function to preprocess the image
def preprocess_image(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # converting from BGR colour scheme to RGB colour scheme
    img = cv.resize(img, (128, 128))  # Resize to 128x128 pixels
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Test the model with a new image
img_path = r"C:\Users\ronar\Desktop\AI-Based Skin Disease Diagnosis Tool\Dataset\archive\skin-disease-datasaet\test_set\VI-chickenpox\125_VI-chickenpox (3).jpg"
img = preprocess_image(img_path)

'''
# show the image uploaded
plt.imshow(cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB))
plt.show()
'''
prediction = model.predict(img)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')
