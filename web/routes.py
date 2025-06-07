import os
import cv2
import shutil
import json
from flask import Blueprint, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from app.model import load_model, predict_image
from app.extract_features import extract_color_features
from app.convert_to_images import convert_pdf_to_images, convert_docx_to_images

web_app = Blueprint("web_app", __name__)

# Path konfigurasi
MODEL_PATH = "model/rf_model.pkl"
MODEL_INFO_PATH = "model/model_info.json"
UPLOAD_FOLDER = "web/static/uploads"
DATASET_FOLDER = "dataset"

# Inisialisasi
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = load_model(MODEL_PATH)

# Label dan harga cetak
label_map = {0: "Hitam Putih", 1: "Warna Sedikit", 2: "Warna Banyak"}
price_map = {"Hitam Putih": 500, "Warna Sedikit": 750, "Warna Banyak": 1000}


@web_app.route("/")
def index():
    return render_template("index.html")


def guess_true_label(features):
    """Mencari label paling mirip berdasarkan dataset yang sudah ada."""
    similarity_scores = {label: [] for label in label_map.values()}
    folder_label_map = {
        "0_hitam_putih": "Hitam Putih",
        "1_warna_sedikit": "Warna Sedikit",
        "2_warna_banyak": "Warna Banyak"
    }

    for folder, label in folder_label_map.items():
        folder_path = os.path.join(DATASET_FOLDER, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            img_path = os.path.join(folder_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            feat = extract_color_features(img)
            score = 1 - cv2.norm(features, feat) / (cv2.norm(features) + 1e-8)
            similarity_scores[label].append(score)

    average_scores = {
        label: sum(scores) / len(scores)
        for label, scores in similarity_scores.items() if scores
    }

    return max(average_scores, key=average_scores.get) if average_scores else None


def load_model_info():
    """Membaca informasi model dari file JSON."""
    try:
        with open(MODEL_INFO_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"❗ Gagal membaca model_info.json: {e}")
        return {"model": "Random Forest", "n_estimators": 100, "accuracy": 0.0}


@web_app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        return "Tidak ada file yang dipilih", 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower()
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    temp_dir = os.path.join(UPLOAD_FOLDER, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Konversi jika perlu
    if ext == "pdf":
        image_paths = convert_pdf_to_images(filepath, temp_dir)
    elif ext == "docx":
        image_paths = convert_docx_to_images(filepath, temp_dir)
    else:
        image_paths = [filepath]

    predictions = []
    correct_count = 0

    count_by_label = {label: 0 for label in label_map.values()}
    accuracy_by_label = {
        label: {"correct": 0, "total": 0} for label in label_map.values()
    }

    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue

        features = extract_color_features(image)
        label_index = predict_image(model, features)
        predicted_label = label_map[label_index]

        true_label = guess_true_label(features)
        is_correct = predicted_label == true_label
        if is_correct:
            correct_count += 1

        predictions.append({
            "filename": os.path.basename(img_path),
            "label": predicted_label,
            "price": price_map[predicted_label]
        })

        count_by_label[predicted_label] += 1
        if true_label in accuracy_by_label:
            accuracy_by_label[true_label]["total"] += 1
            if is_correct:
                accuracy_by_label[true_label]["correct"] += 1

    realtime_accuracy = round(
        (correct_count / len(predictions)) * 100, 2) if predictions else 0.0
    model_info = load_model_info()

    summary = {
        "hitam_putih": count_by_label["Hitam Putih"],
        "warna_sedikit": count_by_label["Warna Sedikit"],
        "warna_banyak": count_by_label["Warna Banyak"],
        "total": sum(p["price"] for p in predictions),
        "realtime_accuracy": realtime_accuracy,
        "label_accuracy": {
            label: {
                "correct": data["correct"],
                "total": data["total"],
                "accuracy": round((data["correct"] / data["total"]) * 100, 2)
                if data["total"] else 0.0
            }
            for label, data in accuracy_by_label.items()
        }
    }

    # # Bersihkan folder temp setelah selesai
    # if os.path.exists(temp_dir):
    #     shutil.rmtree(temp_dir)

    return render_template(
        "result.html",
        predictions=predictions,
        summary=summary,
        model_info=model_info
    )


@web_app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "POST":
        file = request.files.get("image")
        category = request.form.get("category")
        if not file or not category:
            return "File atau label tidak valid", 400

        save_path = os.path.join(DATASET_FOLDER, category)
        os.makedirs(save_path, exist_ok=True)
        file.save(os.path.join(save_path, secure_filename(file.filename)))
        return redirect(url_for("web_app.upload_dataset"))

    return render_template("upload.html")
