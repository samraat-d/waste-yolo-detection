from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
from ultralytics import YOLO  # Import YOLO from ultralytics
from huggingface_hub import hf_hub_download

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'outputs_new')
MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'models')

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Download model from Hugging Face Hub
print("Downloading model from Hugging Face Hub...")
model_name = "samraatd/yolov10-waste-detection"
HUGGINGFACE_TOKEN = "hf_...." # Put your token if required
local_model_path = os.path.join(app.config['MODEL_FOLDER'], "best_local.pt")

try:
    if not os.path.exists(app.config['MODEL_FOLDER']):
        os.makedirs(app.config['MODEL_FOLDER'])

    hf_hub_download(repo_id=model_name, filename="best_local.pt", local_dir=app.config['MODEL_FOLDER']) # token=HUGGINGFACE_TOKEN)
    print(f"Model downloaded to {local_model_path}")
    model = YOLO(local_model_path) #load the downloaded model.
except Exception as e:
    print(f"Error downloading or loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('app.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        file.save(input_path)

        process_image(input_path, output_path)

        flash('Image successfully uploaded and processed')
        return render_template('app.html', input_filename=filename, output_filename=output_filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='outputs_new/' + filename), code=301)

def process_image(input_path, output_path):
    if model is None:
        flash('Model failed to load. Please check the console for errors.')
        return

    results = model(input_path)
    result = results[0]
    boxes = result.boxes.xyxy.numpy().astype(int)
    confidences = result.boxes.conf.numpy()
    class_ids = result.boxes.cls.numpy().astype(int)
    class_names = model.names

    color_map = {
        "Aluminium foil": (255, 0, 0),
        "Bottle": (0, 255, 0),
        "Bottle cap": (0, 0, 255),
        "Broken glass": (255, 165, 0),
        "Can": (255, 255, 0),
        "Carton": (0, 255, 255),
        "Cigarette": (128, 0, 128),
        "Cup": (255, 105, 180),
        "Lid": (255, 69, 0),
        "Other litter": (128, 128, 128),
        "Other plastic": (0, 255, 0),
        "Paper": (0, 128, 0),
        "Plastic bag - wrapper": (255, 182, 193),
        "Plastic container": (186, 85, 211),
        "Pop tab": (255, 140, 0),
        "Straw": (72, 61, 139),
        "Styrofoam piece": (0, 255, 255),
        "Unlabeled litter": (169, 169, 169),
    }

    opencv_image = cv2.imread(input_path)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        label = class_names[class_ids[i]].lower()
        confidence = confidences[i]

        color = color_map.get(label, (255, 255, 255))

        print(f"Predicted label: {label}, Confidence: {confidence:.2f}, Assigned color: {color}")

        cv2.rectangle(opencv_image, (x1, y1), (x2, y2), color, 2)

        label_text = f"{label} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(label_text, font, font_scale, 1)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1]
        cv2.putText(opencv_image, label_text, (text_x, text_y), font, font_scale, color, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, opencv_image)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(debug=True)