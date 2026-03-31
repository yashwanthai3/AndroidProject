from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Generate balanced synthetic dataset
X, y = make_classification(
    n_samples=15000,
    n_features=215,
    n_informative=80,
    n_redundant=20,
    n_classes=2,
    weights=[0.5, 0.5],   # Balanced classes
    random_state=42
)

print("Unique labels:", set(y))

# Stratified split (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print("Accuracy:", acc)

os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "saved_models/best_model.pkl")

print("Model retrained and saved.")
