import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Load tokenizer
tokenizer = joblib.load("tokenizer.pkl")
maxlen = 40

# Load TFLite model
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

class NewsRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(news: NewsRequest):
    seq = tokenizer.texts_to_sequences([news.text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding='post')

    pred = model.predict(padded)[0][0]
    label = "Fake" if pred >= 0.5 else "Real"
    confidence = round(float(pred if label == "Fake" else 1 - pred), 2)

    return {"label": label, "confidence": confidence}
