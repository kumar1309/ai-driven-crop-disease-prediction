from flask import Flask, request, jsonify, render_template, send_file
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load pre-trained EfficientNetB0 model
model = EfficientNetB0(weights='imagenet')

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(img_path):
    """Predict the disease from the image using the pre-trained model."""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    predictions = [{"label": label, "description": desc, "probability": float(prob)} for (label, desc, prob) in decoded]

    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "").lower()

    # Predefined crop disease responses
    crop_disease_responses = {
        "white patches": "White patches can be caused by powdery mildew or mealybugs.",
        "rice blast": "Rice blast is a serious rice disease, managed by planting resistant varieties and fungicide application.",
        "potato blight": "Potato blight can be managed with certified seeds and fungicides.",
    }

    general_responses = {
        "hi": "Hello! How can I assist you with crop disease prediction?",
        "bye": "Goodbye! Feel free to return anytime.",
    }

    # Response logic
    response = crop_disease_responses.get(user_message, general_responses.get(user_message, "Sorry, I didn't understand that."))
    # Add YouTube recommendations
    if user_message in crop_disease_responses:
        response += " Here's a helpful YouTube video: https://www.youtube.com/watch?v=example"
        response += " You can also download a PDF guide for more information."

    return jsonify({"response": response})

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"response": "No image uploaded."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"response": "No selected file."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predictions = predict_disease(filepath)
        response = f"Predicted diseases: {predictions[0]['description']} (probability: {predictions[0]['probability']:.2f})"

        # Append video recommendations
        response += " Here's another YouTube video: https://www.youtube.com/watch?v=another_example"

        return jsonify({"response": response})

    return jsonify({"response": "Invalid file type."}), 400

@app.route('/download-pdf')
def download_pdf():
    path_to_pdf = 'static/disease_guide.pdf'
    return send_file(path_to_pdf, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
