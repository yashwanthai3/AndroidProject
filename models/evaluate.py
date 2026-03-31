import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("../dataset/Android_Malware.csv")
label_column = "class"

X = df.drop(label_column, axis=1)
y = df[label_column]

model = joblib.load("saved_models/best_model.pkl")

predictions = model.predict(X)

print("Classification Report:")
print(classification_report(y, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y, predictions))
