import os
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# === CONFIG ===
MODEL_PATH = "fakenews_lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAXLEN = 40

# === LOAD MODEL & TOKENIZER ===
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

print("Loading tokenizer...")
tokenizer = joblib.load(TOKENIZER_PATH)
print("Tokenizer loaded.")

# === FASTAPI SETUP ===
app = FastAPI()

# === INPUT SCHEMA ===
class NewsRequest(BaseModel):
    text: str

# === PREDICTION ROUTE ===
@app.post("/predict")
def predict(news: NewsRequest):
    # Preprocess
    seq = tokenizer.texts_to_sequences([news.text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding='post')

    # Predict
    pred = model.predict(padded)[0][0]
    label = "Real" if pred >= 0.5 else "Fake"
    confidence = round(float(pred if label == "Real" else 1 - pred), 2)

    return {"label": label, "confidence": confidence}
