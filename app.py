from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import mysql.connector
import uuid
import os
import sendgrid
from sendgrid.helpers.mail import Mail
from sendgrid import SendGridAPIClient
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import string
import random
import tensorflow as tf
from functools import wraps
from datetime import datetime, timedelta
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from deepface import DeepFace
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  

'''UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER'''

UPLOAD_FOLDER = 'static/local_areas'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/logos/'
app.config['LOCAL_AREA_FOLDER'] = 'static/local_area/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
captcha_folder = 'static/captcha'

app.secret_key = 'your_secret_key_here'  
import secrets
print(secrets.token_hex(16))  

if not os.path.exists(captcha_folder):
    os.makedirs(captcha_folder)

if not os.path.exists(app.config['LOCAL_AREA_FOLDER']):
    os.makedirs(app.config['LOCAL_AREA_FOLDER'])



def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

ADMIN_EMAIL = "mamidojuvinay310@gmail.com"
ADMIN_PASSWORD = "M@vinay123"


def send_verification_email(to_email, subject, content):
    api_key = 'SG.JDjon9yJS5e_UQOG2Tuv2Q.FdUgkDEQQl94zBFRWtLKgIytFv9dmvVVRoY234eM'
    sg = SendGridAPIClient(api_key)
    message = Mail(
        from_email='pittalasandeep124@gmail.com',
        to_emails=to_email,
        subject=subject,
        html_content=content
    )
    try:
        sg.send(message)
    except Exception as e:
        print(f"Error sending email: {e}")
        flash('Could not send verification email. Please try again.')


def generate_verification_code():
    return str(uuid.uuid4())[:6]

def generate_captcha():
    captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    captcha_img_path = os.path.join('static/captcha', f"{captcha_text}.png")
    img = Image.new('RGB', (200, 80), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    d.text((20, 30), captcha_text, font=font, fill=(0, 0, 0))
    img.save(captcha_img_path)
    
    return captcha_text, captcha_img_path

def get_db_connection():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='vinay'
    )
    return conn


def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()


    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            verification_code VARCHAR(6),
            valid_until DATETIME DEFAULT NULL,
            logo VARCHAR(255) DEFAULT 'static/logos/default_logo.png'
        )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS missing_persons (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        age INT NOT NULL,
        gender ENUM('Male', 'Female') NOT NULL,
        parent_name VARCHAR(100) NOT NULL,
        parent_phone VARCHAR(15) NOT NULL,
        email VARCHAR(100) NOT NULL,
        image_path VARCHAR(255) NOT NULL,
        status ENUM('not match', 'match') DEFAULT 'not match'
    )
""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS found_persons (
        id INT AUTO_INCREMENT PRIMARY KEY,
        age INT NOT NULL,
        gender ENUM('Male', 'Female') NOT NULL,
        email VARCHAR(100) NOT NULL,
        image_path VARCHAR(255) NOT NULL,
        status ENUM('not match', 'match') DEFAULT 'not match'
    )
""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS LocalArea (
        id INT AUTO_INCREMENT PRIMARY KEY,
        age INT NOT NULL,
        gender ENUM('Male', 'Female') NOT NULL,
        place VARCHAR(100) NOT NULL,
        image_path VARCHAR(255) NOT NULL,
        status ENUM('not match', 'match') DEFAULT 'not match'
    )
""")

    conn.commit()
    conn.close()

create_tables()
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session['user_id'] != 'admin':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# CAPTCHA generation
def generate_captcha_image():
    captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    width, height = 200, 60
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('arial.ttf', 40)
    except IOError:
        font = ImageFont.load_default()
    x = 50
    y = 10
    for char in captcha_text:
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.text((x, y), char, font=font, fill=color)
        x += 30
    captcha_image_path = os.path.join(captcha_folder, f"{captcha_text}.png")
    image.save(captcha_image_path)
    return captcha_text, captcha_image_path

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if not session.get('user_id'):
        flash("Please log in first.")
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        
        if username:
            cursor.execute('UPDATE users SET first_name = %s WHERE id = %s', (username, user_id))
            session['username'] = username  # Update session variable
        if email:
            cursor.execute('UPDATE users SET email = %s WHERE id = %s', (email, user_id))
            session['email'] = email  # Update session variable
        
        if 'logo' in request.files:
            logo = request.files['logo']
            if allowed_file(logo.filename):
                filename = secure_filename(logo.filename)
                logo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                logo.save(logo_path)
                logo_db_path = f"static/logos/{filename}"  # Path to save in the database
                cursor.execute('UPDATE users SET logo = %s WHERE id = %s', (logo_db_path, user_id))
                session['logo'] = logo_db_path  # Update session variable
                flash("Logo uploaded successfully!")

        conn.commit()
        flash("Profile updated successfully!")
        return redirect(url_for('profile'))

    # Fetch user details from the database
    cursor.execute("SELECT first_name, email, logo FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user:
        username, email, logo = user
    else:
        flash("User  not found.")
        return redirect(url_for('login'))

    # Ensure the logo path is correct
    if not logo or logo == '':  # If logo is empty, set to default
        logo = 'logos/default_logo.png'

    return render_template('profile.html', username=username, email=email, logo=logo)


def get_db_connection():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='vinay'
    )
    return conn

@app.route('/captcha/<filename>')
def serve_captcha(filename):
    return send_from_directory(captcha_folder, filename)

@app.route('/admin', methods=['GET'])
def admin():
    if not session.get('user_id'):
        flash("Please log in first.")
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch missing persons with name, age, gender, image_path, and status
    cursor.execute("SELECT name, age, gender, image_path, status FROM missing_persons")
    missing_persons = cursor.fetchall()

    # Fetch found persons with age, gender, email, image_path, and status
    cursor.execute("SELECT age, gender, email, image_path, status FROM found_persons")
    found_persons = cursor.fetchall()

    # Fetch local areas with age, gender, place, image_path, and status
    cursor.execute("SELECT age, gender, place, image_path, status FROM LocalArea")
    local_areas = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template("admin.html", 
                          missing_persons=missing_persons, 
                          found_persons=found_persons, 
                          local_areas=local_areas)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash("Passwords do not match!")
            return redirect(url_for('register'))

        verification_code = generate_verification_code()
        valid_until = datetime.now() + timedelta(minutes=5)

        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Email is already registered. Please login.')
            return redirect(url_for('login'))

        cursor.execute('INSERT INTO users (first_name, last_name, email, password, verification_code, valid_until) VALUES (%s, %s, %s, %s, %s, %s)',
            (first_name, last_name, email, hashed_password, verification_code, valid_until)
        )
        conn.commit()
        send_verification_email(email, "Email Verification", f"Your verification code is {verification_code}. It is valid for 5 minutes.")
        flash('Registration successful! Please check your email to verify your account.')
        return redirect(url_for('verify_otp1', email=email))

    return render_template('register.html')



@app.route('/verify_otp1', methods=['GET', 'POST'])
def verify_otp1():
    email = request.args.get('email')

    if request.method == 'POST':
        entered_otp = request.form['otp']

        # Create a database connection
        conn = get_db_connection()

        # Use dictionary=True to return rows as dictionaries
        cursor = conn.cursor(dictionary=True)

        # Query to fetch the user by email and entered OTP
        cursor.execute('SELECT * FROM users WHERE email = %s AND verification_code = %s', (email, entered_otp))
        user = cursor.fetchone()

        if user:
            # Check if valid_until is a string or datetime
            valid_until = user['valid_until']
            if isinstance(valid_until, str):
                
                valid_until = datetime.fromisoformat(valid_until)

            if datetime.now() <= valid_until:
                cursor.execute('UPDATE users SET verification_code = NULL, valid_until = NULL WHERE email = %s', (email,))
                conn.commit()

                flash('Registered successfully! You can now log in.')
                return redirect(url_for('login'))
            else:
                flash('Invalid or expired OTP. Please try again.')
        else:
            flash('Invalid OTP or email not found.')

    return render_template('verify_otp1.html', email=email)



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        entered_captcha = request.form['captcha'].strip()
        
        conn = get_db_connection()

        cursor = conn.cursor(dictionary=True)
        
        
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()

        # Validate the captcha entered by the user
        if 'captcha_text' in session and entered_captcha != session['captcha_text']:
            flash('Invalid captcha. Please try again.')
            return redirect(url_for('login'))
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session['user_id'] = 'admin'  
            return redirect(url_for('admin'))

        # If user is found, check password and verification code
        if user and check_password_hash(user['password'], password) and user['verification_code'] is None:
            session['user_id'] = user['id']  # Store user ID in session
            session['user_name'] = user['first_name']  # Store user first name in session
            flash('Login successful!')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your email and password or verify your email.')
            return redirect(url_for('login'))

    # If it's a GET request, generate the captcha and render the login page
    captcha_text, captcha_img_path = generate_captcha()
    session['captcha_text'] = captcha_text

    return render_template('login.html', captcha_img=captcha_img_path)


@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
    if request.method == 'POST':
        email = request.form['email']

        # Create a database connection
        conn = get_db_connection()

        # Use dictionary=True to return rows as dictionaries
        cursor = conn.cursor(dictionary=True)

        # Query to fetch the user by email
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()

        if user:
            otp = str(uuid.uuid4())[:6]  # Generate a 6-digit OTP
            cursor.execute('UPDATE users SET verification_code = %s, valid_until = %s WHERE email = %s', 
                           (otp, datetime.now() + timedelta(minutes=5), email))
            conn.commit()

            # Send the OTP to the user's email
            send_verification_email(email, 'Reset Your Password', f'Your OTP is: {otp}. Please verify it.')
            flash('OTP sent to your email.')
            return redirect(url_for('verify_otp2', email=email))
        else:
            flash('Email not found.')

        return redirect(url_for('forgot'))

    return render_template('forgot.html')


@app.route('/verify_otp2', methods=['GET', 'POST'])
def verify_otp2():
    email = request.args.get('email')
    if request.method == 'POST':
        entered_otp = request.form['otp']

        # Create a database connection
        conn = get_db_connection()

        # Use dictionary=True to return rows as dictionaries
        cursor = conn.cursor(dictionary=True)

        # Query to fetch the user by email and entered OTP
        cursor.execute('SELECT * FROM users WHERE email = %s AND verification_code = %s', (email, entered_otp))
        user = cursor.fetchone()

        if user:
            # Check if valid_until is a string or datetime
            valid_until = user['valid_until']
            if isinstance(valid_until, str):
                # If it's a string, parse it as ISO format
                valid_until = datetime.fromisoformat(valid_until)

            if datetime.now() <= valid_until:
                # Clear the verification code and valid_until fields after verification
                cursor.execute('UPDATE users SET verification_code = NULL, valid_until = NULL WHERE email = %s', (email,))
                conn.commit()

                flash('OTP verified! You can now reset your password.')
                return redirect(url_for('reset_password', email=email))
            else:
                flash('Invalid or expired OTP.')
        else:
            flash('Invalid OTP or email not found.')

    return render_template('verify_otp2.html', email=email)



# Reset password route
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    email = request.args.get('email')
    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('reset_password', email=email))

        conn = get_db_connection()
        cursor = conn.cursor()
        # Use %s for MySQL query parameter placeholders
        cursor.execute('UPDATE users SET password = %s WHERE email = %s', (generate_password_hash(password), email))
        conn.commit()

        flash('Password reset successfully! You can now log in.')
        return redirect(url_for('login'))

    return render_template('reset_password.html', email=email)





class ImageComparison:
    def __init__(self):
        
        self.efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.efficientnet_model = tf.keras.models.Model(inputs=self.efficientnet_model.input, 
                                                      outputs=self.efficientnet_model.output)

    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for EfficientNetB0 or DeepFace"""
        try:
            img = image.load_img(image_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def get_efficientnet_features(self, image_path):
        """Extract features using EfficientNetB0"""
        img = self.preprocess_image(image_path)
        if img is None:
            return None
        features = self.efficientnet_model.predict(img)
        return features.flatten()

    def get_deepface_features(self, image_path):
        """Extract facial embeddings using DeepFace"""
        try:
            embedding = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=False)
            return np.array(embedding[0]['embedding'])
        except Exception as e:
            print(f"DeepFace error: {e}")
            return None

    def compare_images(self, image1_path, image2_path):
        """Compare two images using a combination of methods"""
        # Extract features
        eff_features1 = self.get_efficientnet_features(image1_path)
        eff_features2 = self.get_efficientnet_features(image2_path)
        
        deepface_features1 = self.get_deepface_features(image1_path)
        deepface_features2 = self.get_deepface_features(image2_path)

        # Check if feature extraction failed
        if eff_features1 is None or eff_features2 is None:
            return False

        # Compute similarities
        results = {}

        # 1. EfficientNet Cosine Similarity
        results['efficientnet_cosine'] = cosine_similarity([eff_features1], [eff_features2])[0][0]

        # 2. DeepFace Cosine Similarity (if facial features are detected)
        if deepface_features1 is not None and deepface_features2 is not None:
            results['deepface_cosine'] = cosine_similarity([deepface_features1], [deepface_features2])[0][0]
        else:
            results['deepface_cosine'] = None

        # Decision logic with thresholds
        thresholds = {
            'efficientnet_cosine': 0.85,
            'deepface_cosine': 0.7  # Updated threshold for DeepFace
        }

        match_criteria = [
            results['efficientnet_cosine'] > thresholds['efficientnet_cosine']
        ]

        # If DeepFace is applicable, add it to criteria
        if results['deepface_cosine'] is not None:
            match_criteria.append(results['deepface_cosine'] > thresholds['deepface_cosine'])

        # Determine if there is a match
        is_match = any(match_criteria)

        return is_match



@app.route('/missing_person', methods=['GET', 'POST'])
def missing_person():
    if request.method == 'POST':
        name = request.form["name"]
        age = request.form["age"]
        gender = request.form["gender"]
        parent_name = request.form["parent_name"]
        parent_phone = request.form["parent_phone"]
        email = request.form["email"]
        
        image_file = request.files["image"]
        image_path = os.path.join("static/missing", secure_filename(image_file.filename))
        image_file.save(image_path)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO missing_persons (name, age, gender, parent_name, parent_phone, email, image_path, status) VALUES (%s, %s, %s, %s, %s, %s, %s, 'not match')",
            (name, age, gender, parent_name, parent_phone, email, image_path)
        )
        conn.commit()

        # Check only 'not match' records in found_persons
        cursor.execute("SELECT * FROM found_persons WHERE gender = %s AND status = 'not match'", (gender,))
        found_persons = cursor.fetchall()
        
        matched_person = None
        image_comparator = ImageComparison()
        for found_person in found_persons:
            if len(found_person) > 4:
                if image_comparator.compare_images(image_path, found_person[4]):
                    matched_person = found_person
                    # Update status to 'match' for both missing and found person
                    cursor.execute("UPDATE missing_persons SET status = 'match' WHERE id = %s", (cursor.lastrowid,))
                    cursor.execute("UPDATE found_persons SET status = 'match' WHERE id = %s", (found_person[0],))
                    conn.commit()
                    break

        if matched_person:
            send_verification_email(email, "Match Found", f"Details for the found person: {matched_person[1]}, {matched_person[3]}.")
            send_verification_email(matched_person[3], "Match Found", f"Details for the missing person: {name}, {email}.")
            flash("Match found and contact details exchanged!")
        else:
            flash("No match found in found persons, but your missing person report has been saved.")
        
        # Check only 'not match' records in LocalArea
        cursor.execute("SELECT * FROM LocalArea WHERE gender = %s AND status = 'not match'", (gender,))
        local_areas = cursor.fetchall()
        
        for local_area in local_areas:
            if len(local_area) > 4:
                if image_comparator.compare_images(image_path, local_area[4]):
                    matched_person = local_area
                    # Update status to 'match' for both missing and local area
                    cursor.execute("UPDATE missing_persons SET status = 'match' WHERE id = %s", (cursor.lastrowid,))
                    cursor.execute("UPDATE LocalArea SET status = 'match' WHERE id = %s", (local_area[0],))
                    conn.commit()
                    break

        if matched_person:
            send_verification_email(email, "Match Found in Local Area", f"Details for the found person: {matched_person[1]}, {matched_person[3]}.")
            send_verification_email(matched_person[3], "Match Found in Local Area", f"Details for the missing person: {name}, {email}.")
            flash("Match found in local area and contact details exchanged!")
        else:
            flash("No match found from Local Areas, but your missing person report has been saved.")

        cursor.close()
        conn.close()
        return redirect(url_for("home"))

    return render_template("missing_person.html", user_id=session.get('user_id'))

@app.route('/found_person', methods=['GET', 'POST'])
def found_person():
    if request.method == 'POST':
        if not session.get('user_id'):
            flash("Please log in first.")
            return redirect(url_for('login'))
        age = request.form["age"]
        gender = request.form["gender"]
        email = request.form["email"]
        image_file = request.files["image"]
        image_path = os.path.join("static/found", secure_filename(image_file.filename))
        image_file.save(image_path)

        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO found_persons (age, gender, email, image_path, status) VALUES (%s, %s, %s, %s, 'not match')",
            (age, gender, email, image_path)
        )
        conn.commit()

        # Check only 'not match' records in missing_persons
        cursor.execute("SELECT * FROM missing_persons WHERE gender = %s AND age = %s AND status = 'not match'", (gender, age))
        missing_persons = cursor.fetchall()

        matched_person = None
        image_comparator = ImageComparison()
        for missing_person in missing_persons:
            if image_comparator.compare_images(image_path, missing_person[7]):  # Adjusted index for status column
                matched_person = missing_person
                # Update status to 'match' for both found and missing person
                cursor.execute("UPDATE found_persons SET status = 'match' WHERE id = %s", (cursor.lastrowid,))
                cursor.execute("UPDATE missing_persons SET status = 'match' WHERE id = %s", (missing_person[0],))
                conn.commit()
                break

        if matched_person:
            send_verification_email(missing_person[5], "Match Found", f"Details for the found person: Age: {age}, Gender: {gender}.")
            send_verification_email(email, "Match Found", f"Details for the missing person: Name: {matched_person[1]}, Age: {matched_person[2]}, Gender: {matched_person[3]}.")
            flash("Match found and contact details exchanged!")
        else:
            flash("No match found; your found person report has been saved!")

        cursor.close()
        conn.close()
        return redirect(url_for("home"))

    return render_template("found_person.html", user_id=session.get('user_id'))

@app.route('/add_local_area', methods=['GET', 'POST'])
def add_local_area():
    if request.method == 'POST':
        age = request.form.get("age")
        gender = request.form.get("gender")
        place = request.form.get("place")
        image_file = request.files.get("image")

        if not age or not gender or not place or not image_file:
            flash("Please fill in all fields.")
            return redirect(url_for("add_local_area"))

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        image_file.save(image_path)

        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO LocalArea (age, gender, place, image_path, status) VALUES (%s, %s, %s, %s, 'not match')",
            (age, gender, place, image_path)
        )
        conn.commit()

        # Check only 'not match' records in missing_persons
        cursor.execute("SELECT * FROM missing_persons WHERE gender = %s AND status = 'not match'", (gender,))
        missing_persons = cursor.fetchall()
        
        matched_person = None
        image_comparator = ImageComparison()
        for missing_person in missing_persons:
            if len(missing_person) > 7:  # Adjusted index for status column
                missing_image_path = missing_person[6]
                if os.path.exists(missing_image_path):
                    if image_comparator.compare_images(image_path, missing_image_path):
                        matched_person = missing_person
                        # Update status to 'match' for both local area and missing person
                        cursor.execute("UPDATE LocalArea SET status = 'match' WHERE id = %s", (cursor.lastrowid,))
                        cursor.execute("UPDATE missing_persons SET status = 'match' WHERE id = %s", (missing_person[0],))
                        conn.commit()
                        break

        if matched_person:
            send_verification_email(matched_person[5], "Match Found in Local Area", f"Details for the local area: at {place}, Age: {age}.")
            send_verification_email(matched_person[5], "Match Found in Local Area", f"Your child details: Name: {matched_person[1]}, Age: {missing_person[2]}.")
            flash("Match found and contact details exchanged!")
        else:
            flash("No match found in missing persons, but your local area report has been saved.")

        cursor.close()
        conn.close()
        flash("Local area added successfully!")
        return redirect(url_for("add_local_area"))

    return render_template("add_local_area.html")

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out.")
    return redirect(url_for('home'))

if __name__ == '__main__':

    app.run(debug=True)
