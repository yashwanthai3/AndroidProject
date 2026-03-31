from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import numpy as np
from backend.feature_extractor import extract_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "ml_models", "saved_models", "best_model.pkl")

model = joblib.load(model_path)

@app.post("/predict-apk")
async def predict_apk(file: UploadFile = File(...)):

    temp_path = "temp.apk"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    features = extract_features(temp_path)

    prediction = model.predict(features.reshape(1, -1))[0]

    result = "Malware" if prediction == 1 else "Benign"

    return {"prediction": result}