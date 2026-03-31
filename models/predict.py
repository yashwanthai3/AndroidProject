import joblib
import pandas as pd

model = joblib.load("saved_models/best_model.pkl")

def predict_from_features(feature_list):
    prediction = model.predict([feature_list])
    return int(prediction[0])
