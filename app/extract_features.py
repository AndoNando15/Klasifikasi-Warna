
# app/extract_features.py
import numpy as np
import cv2

def extract_color_features(image):
    image = cv2.resize(image, (100, 100))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    return hist.flatten()
