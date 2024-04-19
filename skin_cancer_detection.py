
import os
from flask import Flask, request, render_template, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Initialize Flask app
app = Flask(__name__)

# Set allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define uploads directory
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load trained model
model = load_model(r'D:\annaaa\skin_cancer_detection_using_deeplearning\templates\Skin cancer.hdf5')

# Define labels for skin cancer types
cancer_types = {
    0: 'Melanocytic nevi',
    1: 'Melanoma-Cancerous',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma-Cancerous',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma',
    7: 'Normal skin'
}

# Function to predict and assign risk
def predict_and_assign_risk(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    result = model.predict(x)
    
    # Classify into benign and malignant
    benign_prob = result[0][0]
    if benign_prob >= 0.5:
        risk_level = 'Benign'
        cancer_type_index = np.argmax(result[0][1:]) + 2  # Start from index 2 for cancer types
    else:
        risk_level = 'Malignant'
        cancer_type_index = np.argmax(result[0][1:]) + 1  # Start from index 1 for cancer types
    
    # Get cancer type
    cancer_type = cancer_types[cancer_type_index]
    
    return risk_level, cancer_type

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            risk_level, cancer_type = predict_and_assign_risk(file_path)
            return render_template('result.html', filename=filename, risk_level=risk_level, cancer_type=cancer_type)
    return redirect(request.url)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()



with open('templates/index.html', 'w') as file:
    file.write(index_html)

with open('templates/result.html', 'w') as file:
    file.write(result_html)
