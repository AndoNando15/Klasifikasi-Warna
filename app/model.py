
# app/model.py
import joblib

def load_model(path):
    return joblib.load(path)

def predict_image(model, features):
    return model.predict([features])[0]

