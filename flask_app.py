from flask import Flask, render_template, request, flash
import cv2
from model_inference import KidneyStoneDetectionModel
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# Load the model globally when the app starts
model = KidneyStoneDetectionModel(model_path="./ks_detection.pt")
print("Model loaded successfully.")

@app.route('/')
def home():
    return render_template("Login.html")

@app.route("/Login", methods=['GET', 'POST'])
def adminlogin():
    if request.method == 'POST':
        if request.form['uname'] == 'admin' and request.form['password'] == 'admin':
            print(request.form['uname'])
            flash("Login successfully")
            return render_template('index.html')

        else:
            flash("UserName Or Password Incorrect!")
            return render_template('Login.html')

@app.route('/index', methods=["GET", "POST"])
def index():
    global original_image_path, processed_image_path, processed_image, severity

    if request.method == 'POST':
        # Check if the file part is present in the request
        if 'uploadedImage' not in request.files:
            return "No file part in the request.", 400

        # Get the file from the POST request
        original_image = request.files["uploadedImage"]

        if original_image.filename == "":
            return "No file selected.", 400

        # Define paths for saving uploaded and processed images
        original_image_path = "static/uploaded_image.jpg"
        original_image.save(original_image_path)

        # Read and convert the image to the required format (RGB)
        original_image_np = cv2.imread(original_image_path)
        if original_image_np is None:
            return "Error loading the image. Please try again.", 500

        original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_BGR2RGB)

        # Run kidney stone detection
        try:
            num_stones, max_stone_size = model.run_inference(image=original_image_np)
            processed_image, severity, stone_sizes = model.annotate_image_with_sizes(image=original_image_np)
            
            processed_image_path = "static/processed_image.jpg"

            # Save the processed image
            plt.imsave(processed_image_path, processed_image)

            # If no stones are detected, set a "no stones" message and consultation
            no_stone_message = None
            consultation_text = ""
            if num_stones == 0:
                no_stone_message = "No kidney stones detected in the uploaded image."
                consultation_text = ("There are no kidney stones detected. "
                                     "It is advisable to maintain a healthy lifestyle and stay hydrated. "
                                     "If you experience any symptoms, please consult a healthcare provider.")
            else:
                no_stone_message = "Kidney stones detected in the uploaded image."
                # Determine consultation text based on stone size
                consultation_text = generate_consultation_text(max_stone_size)

        except Exception as e:
            return f"Error during model inference: {str(e)}", 500

        return render_template("index.html", img_paths=[original_image_path, processed_image_path],
                               severity=severity, consultation_text=consultation_text,
                               no_stone_message=no_stone_message, stone_sizes=stone_sizes)

    return render_template("index.html")


def generate_consultation_text(max_stone_size):
    """
    Generate consultation text based on stone size.
    """

    if max_stone_size < 50:
        return ("Your kidney stone is small (less than 5 mm). "
                "It is recommended to drink plenty of water (at least 2-3 liters daily) "
                "to help pass the stone naturally. Over-the-counter pain relievers can help manage any discomfort. "
                "You may consider using Tamsulosin to assist with stone passage.")
    elif 50 <= max_stone_size <= 100:
        return ("Your kidney stone is moderate in size (5 mm to 10 mm). "
                "It is crucial to continue hydrating and monitoring the situation. "
                "Dietary changes, such as reducing oxalate-rich foods, are advised. "
                "You may need medications to facilitate stone passage and manage pain.")
    else:
        return ("Your kidney stone is large (greater than 10 mm). "
                "Surgical intervention may be necessary. It is essential to consult a urologist for evaluation. "
                "Pain management and antibiotics may be prescribed if there are complications.")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)
