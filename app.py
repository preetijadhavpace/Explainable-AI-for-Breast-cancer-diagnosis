import random
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mail import Mail, Message  
from tensorflow.keras.models import load_model
import os
import numpy as np
import tensorflow as tf
from utils import preprocess_image, make_gradcam_heatmap, save_and_overlay_heatmap
from database import get_db_connection, init_db
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'super_secret_key_123'

# ✅ Flask-Mail Configuration (update with your email)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'preetisj711@gmail.com'      # 🔹 Replace with your email
app.config['MAIL_PASSWORD'] = 'uytl vabd zlld owjs'         # 🔹 Replace with app password (not normal password)
app.config['MAIL_DEFAULT_SENDER'] = ('Breast Cancer XAI', 'your_email@gmail.com')

mail = Mail(app)

# Ensure DB exists
init_db()

# Model setup
MODEL_PATH = os.path.join('model', 'densenet201_model.h5')
model = load_model(MODEL_PATH)

CLASS_NAMES = ['Benign', 'Malignant']
LAST_CONV_LAYER = 'conv5_block32_concat'


# -----------------------------
# Generate Explanation Function
# -----------------------------
def generate_explanation(label, confidence, activation_level):

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
            f"🩸 Observed Pattern: {pattern}\n"
            f"📊 Estimated Stage: {stage}\n\n"
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
            f"Observed Pattern: {pattern}\n"
            f"The Grad-CAM average activation ({0.0 if np.isnan(activation_level) else activation_level:.2f}) suggests "
            "low model attention overall, typical for non-cancerous mammograms."
        )

    return explanation


# -----------------------------
# ROUTES
# -----------------------------

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['GET'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', label=None)


@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('upload'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('upload'))

    upload_folder = app.config.get('UPLOAD_FOLDER', os.path.join('static', 'uploads'))
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)


    x = preprocess_image(filepath)

    try:
        preds = model.predict({"input_layer": x})
    except:
        preds = model.predict(x)

    pred_class = CLASS_NAMES[int(np.argmax(preds))]
    confidence = float(np.max(preds) * 100)


    heatmap, activation_level = make_gradcam_heatmap(x, model, LAST_CONV_LAYER)

    heatmap_path = save_and_overlay_heatmap(filepath, heatmap)
    
    explanation = generate_explanation(pred_class, confidence, activation_level)

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (user_id, filename, label, confidence, explanation, heatmap_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session['user_id'], file.filename, pred_class, confidence, explanation, heatmap_path))
    conn.commit()
    conn.close()

    session.pop('_flashes', None)
    return render_template(
        'result.html',
        image_file=filepath,
        heatmap_file=heatmap_path,
        label=pred_class,
        confidence=round(confidence, 2),
        explanation=explanation
    )


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT name, email, specialization FROM radiologists WHERE id=?", (session['user_id'],))
    user = c.fetchone()
    c.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY timestamp DESC", (session['user_id'],))
    results = c.fetchall()
    conn.close()

    return render_template('dashboard.html', username=session['username'], user=user, results=results)


# ✅ REGISTER (with email OTP)
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        specialization = request.form['specialization']

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM radiologists WHERE email=?", (email,))
        existing_user = c.fetchone()

        if existing_user:
            flash('User already exists. Please log in.', 'warning')
            conn.close()
            return redirect(url_for('login'))

        hashed_password = generate_password_hash(password)
        otp = str(random.randint(100000, 999999))

        c.execute('''
            INSERT INTO radiologists (name, email, password, specialization, verification_code)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, email, hashed_password, specialization, otp))
        conn.commit()
        conn.close()

        # ✅ Send OTP email
        try:
            msg = Message('Your OTP Verification Code - Breast Cancer XAI', recipients=[email])
            msg.body = f'Hello {name},\n\nYour verification OTP code is: {otp}\n\nThank you for registering!\n- Breast Cancer XAI Team'
            mail.send(msg)
            flash('OTP sent to your registered email. Please verify to activate your account.', 'info')
        except Exception as e:
            print("❌ Error sending email:", e)
            flash('Error sending OTP email. Please check configuration.', 'danger')

        return redirect(url_for('verify'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM radiologists WHERE email=?", (email,))
        user = c.fetchone()
        conn.close()

        if user:
            if user['is_verified'] == 0:
                flash('Please verify your account first using the OTP sent to your email.', 'warning')
                return redirect(url_for('verify'))

            if check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['name']
                flash('Logged in successfully!', 'success')
                return redirect(url_for('upload'))
            else:
                flash('Incorrect password. Try again.', 'danger')
                return render_template('login.html')
        else:
            flash('No account found with that email. Please register first.', 'danger')
            return render_template('login.html')

    return render_template('login.html')


@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        email = request.form['email']
        otp_entered = request.form['otp']

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM radiologists WHERE email=? AND verification_code=?", (email, otp_entered))
        user = c.fetchone()

        if user:
            c.execute("UPDATE radiologists SET is_verified=1, verification_code=NULL WHERE email=?", (email,))
            conn.commit()
            conn.close()
            session.pop('_flashes', None)
            flash('Account verified successfully! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid OTP or email. Try again.', 'danger')
            return render_template('verify.html')

    return render_template('verify.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/index')
def index():
    return render_template('index.html')



@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(f"\n📩 New message received:\nFrom: {name} ({email})\nMessage: {message}\n")
        flash('Message sent successfully!', 'success')
        return render_template('contact.html')
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
