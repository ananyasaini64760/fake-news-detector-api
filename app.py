from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("fakenews_lstm_model.h5")
tokenizer = joblib.load("tokenizer.pkl")
maxlen = 40

app = FastAPI()

class NewsRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(news: NewsRequest):
    seq = tokenizer.texts_to_sequences([news.text])
    padded = pad_sequences(seq, maxlen=maxlen, padding='post')
    pred = model.predict(padded)[0][0]
    label = "Real" if pred >= 0.5 else "Fake"
    confidence = round(float(pred if label == "Real" else 1 - pred), 2)
    return {"label": label, "confidence": confidence}
