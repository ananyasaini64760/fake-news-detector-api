import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

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
    padded = pad_sequences(seq, maxlen=maxlen, padding='post')

    # Run model
    interpreter.set_tensor(input_index, np.array(padded, dtype=np.float32))
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)[0][0]

    label = "Real" if pred >= 0.5 else "Fake"
    confidence = round(float(pred if label == "Real" else 1 - pred), 2)
    return {"label": label, "confidence": confidence}
