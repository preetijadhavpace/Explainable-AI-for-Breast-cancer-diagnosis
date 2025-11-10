import random
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
import os
import numpy as np
from utils import preprocess_image, make_gradcam_heatmap, save_and_overlay_heatmap
import tensorflow as tf

# Initialize Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your_secret_key_here'  # Required for sessions

# Folder for uploaded images
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Load model
MODEL_PATH = os.path.join('model', 'densenet201_model.h5')
model = load_model(MODEL_PATH)

CLASS_NAMES = ['Benign', 'Malignant']
LAST_CONV_LAYER = 'conv5_block32_concat'  # update if needed



def generate_explanation(label, confidence, activation_level):
    """Explain the prediction based on Grad-CAM activation strength."""
    if label == 'Malignant':
        if activation_level >= 0.6:
            pattern = "widespread abnormal tissue density"
        elif activation_level >= 0.3:
            pattern = "localized irregular tissue structure"
        else:
            pattern = "small potential dense region"

        if confidence >= 90:
            stage = "Possible Advanced Stage (Stage III)"
        elif confidence >= 75:
            stage = "Likely Intermediate Stage (Stage II)"
        else:
            stage = "Likely Early Stage (Stage I)"

        explanation = (
            f"The model predicts this mammogram as malignant with {confidence:.2f}% confidence.\n\n"
            f"🩸 **Observed Pattern:** {pattern}\n"
            f"📊 **Estimated Stage:** {stage}\n\n"
            f"The Grad-CAM heatmap shows stronger activations (average intensity: {activation_level:.2f}), "
            "indicating that the model focused on tissue regions exhibiting structural irregularities "
            "consistent with malignant features."
        )

    else:
        if activation_level < 0.15:
            pattern = "no abnormality — uniformly low activation across the scan."
        elif activation_level < 0.3:
            pattern = "minor local tissue density, but no malignant indicators."
        else:
            pattern = "slightly active zones possibly due to benign dense tissue."

        explanation = (
            f"The model predicts this mammogram as benign with {confidence:.2f}% confidence.\n\n"
            f" Observed Pattern:  {pattern}\n"
            f"The Grad-CAM average activation ({0.0 if np.isnan(activation_level) else activation_level:.2f}) suggests\n"
           
            "low model attention overall, typical for non-cancerous mammograms."
        )

    return explanation







@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Simple fixed login (you can replace this with DB later)
        if username == 'admin' and password == '1234':
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        user = session['user']
    else:
        user = "Guest"
    return render_template('index.html', user=user)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))



@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Save uploaded image
    upload_folder = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    # Preprocess for model input
    x = preprocess_image(filepath)

    # Predict class
    preds = model.predict(x)
    pred_class = CLASS_NAMES[int(np.argmax(preds))]
    confidence = float(np.max(preds) * 100)

    # Generate Grad-CAM heatmap
    heatmap, activation_level = make_gradcam_heatmap(x, model, LAST_CONV_LAYER)

    heatmap_path = save_and_overlay_heatmap(filepath, heatmap)

    # Generate textual explanation
    explanation = generate_explanation(pred_class, confidence, activation_level)


    # Render result page
    return render_template(
        'result.html',
        image_file=filepath,
        heatmap_file=heatmap_path,
        label=pred_class,
        confidence=round(confidence, 2),
        explanation=explanation
    )


if __name__ == '__main__':
    app.run(debug=True)
