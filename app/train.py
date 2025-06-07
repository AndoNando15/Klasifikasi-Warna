import os
import cv2
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
from app.extract_features import extract_color_features

# Konfigurasi
DATASET_DIR = "dataset/"
OUTPUT_MODEL = "model/rf_model.pkl"
OUTPUT_INFO = "model/model_info.json"
LABELS = ["0_hitam_putih", "1_warna_sedikit", "2_warna_banyak"]

# Siapkan data
features = []
labels = []

for idx, label_dir in enumerate(LABELS):
    path = os.path.join(DATASET_DIR, label_dir)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            feat = extract_color_features(img)
            features.append(feat)
            labels.append(idx)

# Training model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(features, labels)

# Simpan model
os.makedirs("model", exist_ok=True)
joblib.dump(clf, OUTPUT_MODEL)
print("✅ Model trained and saved to:", OUTPUT_MODEL)

# Evaluasi dengan cross-validation (5-fold)
scores = cross_val_score(clf, features, labels, cv=5)
accuracy = round(np.mean(scores) * 100, 2)
print(f"✅ Cross-validated Accuracy: {accuracy}%")

# Simpan info model
model_info = {
    "model": "Random Forest",
    "n_estimators": 100,
    "accuracy": accuracy
}

with open(OUTPUT_INFO, "w") as f:
    json.dump(model_info, f)

print("✅ Model info saved to:", OUTPUT_INFO)
