from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Folder for uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels
labels = ['Benign', 'Malignant']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result='No image selected')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Preprocess image
        img = image.load_img(filepath, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        # Predict
        prediction = model.predict(x)[0]
        predicted_class = labels[int(np.argmax(prediction))]

        # Return only class name (no duplicate 'Prediction:' text)
        return render_template('index.html',
                               result=predicted_class,
                               image_path=filepath)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
