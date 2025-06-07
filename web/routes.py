import os
import cv2
from flask import Blueprint, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from app.model import load_model, predict_image
from app.extract_features import extract_color_features
from app.convert_to_images import (
    convert_pdf_to_images,
    convert_docx_to_images
    # ,
    # delete_temp_folder  # pastikan ini ada di convert_to_images.py
)

web_app = Blueprint('web_app', __name__)

MODEL_PATH = "model/rf_model.pkl"
UPLOAD_FOLDER = "web/static/uploads"
DATASET_FOLDER = "dataset"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)

@web_app.route("/")
def index():
    return render_template("index.html")


@web_app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    if not file or file.filename == "":
        return "Tidak ada file yang dipilih", 400

    true_label = request.form.get("true_label")
    filename = secure_filename(file.filename)
    file_ext = filename.lower().split('.')[-1]
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    temp_dir = os.path.join(UPLOAD_FOLDER, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    if file_ext == "pdf":
        image_paths = convert_pdf_to_images(filepath, temp_dir)
    elif file_ext == "docx":
        image_paths = convert_docx_to_images(filepath, temp_dir)
    else:
        image_paths = [filepath]

    predictions = []
    correct_count = 0

    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        features = extract_color_features(image)
        label = predict_image(model, features)

        label_map = {0: "Hitam Putih", 1: "Warna Sedikit", 2: "Warna Banyak"}
        price_map = {0: "500", 1: "750", 2: "1000"}

        predicted_label = label_map[label]
        if predicted_label == true_label:
            correct_count += 1

        predictions.append({
            "filename": os.path.basename(img_path),
            "label": predicted_label,
            "price": price_map[label]
        })

    realtime_accuracy = round((correct_count / len(predictions)) * 100, 2) if predictions else 0

    summary = {
        "hitam_putih": sum(1 for p in predictions if p["label"] == "Hitam Putih"),
        "warna_sedikit": sum(1 for p in predictions if p["label"] == "Warna Sedikit"),
        "warna_banyak": sum(1 for p in predictions if p["label"] == "Warna Banyak"),
        "total": sum(int(p["price"]) for p in predictions),
        "realtime_accuracy": realtime_accuracy,
        "true_label": true_label
    }

    # Auto-hapus folder temp (jika ingin aktifkan lagi)
    # delete_temp_folder(temp_dir)

    return render_template("result.html", predictions=predictions, summary=summary)


@web_app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "POST":
        file = request.files['image']
        category = request.form['category']
        save_path = os.path.join(DATASET_FOLDER, category)
        os.makedirs(save_path, exist_ok=True)
        file.save(os.path.join(save_path, secure_filename(file.filename)))
        return redirect(url_for("web_app.upload_dataset"))

    return render_template("upload.html")
