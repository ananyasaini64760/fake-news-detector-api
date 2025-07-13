import os
import requests
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = FastAPI()
model = load_model("fakenews_lstm_model.h5")
# === SETTINGS ===
MODEL_PATH = "model.tflite"
TOKENIZER_PATH = "tokenizer.pkl"
MAXLEN = 40

# === DOWNLOAD TFLITE MODEL IF NOT PRESENT ===
def download_tflite_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model.tflite...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        response = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete.")

# === SETUP ===
download_tflite_model()
tokenizer = joblib.load(TOKENIZER_PATH)

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_index = interpreter.get


