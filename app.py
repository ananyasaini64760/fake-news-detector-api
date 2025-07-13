import os
import gdown
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Suppress TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# === CONFIG ===
MODEL_PATH = "fakenews_lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAXLEN = 40
GOOGLE_DRIVE_FILE_ID = "https://drive.google.com/file/d/1OeFOjCVDq_MrDUUGyAWqWhfPNrepi8zM/view?usp=sharing"  # 🔁 Replace with your file ID

# === AUTO-DOWNLOAD MODEL FROM GOOGLE DRIVE ===
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️  Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model download complete.")

download_model()

# === LOAD MODEL & TOKENIZER ===
print("🔁 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")

print("🔁 Loading tokenizer...")
tokenizer = joblib.load(TOKENIZER_PATH)
print("✅ Tokenizer loaded.")

# === FASTAPI SETUP ===
app = FastAPI()

# === INPUT SCHEMA ===
class NewsRequest(BaseModel):
    text: str

# === PREDICTION ENDPOINT ===
@app.post("/predict")
def predict(news: NewsRequest):
    seq = tokenizer.texts_to_sequences([news.text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding='post')

    pred = model.predict(padded)[0][0]
    label = "Real" if pred >= 0.5 else "Fake"
    confidence = round(float(pred if label == "Real" else 1 - pred), 2)

    return {"label": label, "confidence": confidence}
