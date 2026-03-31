import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Training XGBoost Model...")

# -----------------------------
# STEP 1: Create Dataset (5 features for APK permissions)
# -----------------------------
X = np.array([
    [0,0,0,0,0],  # benign
    [0,0,0,0,1],  # benign
    [0,0,1,0,0],  # benign
    [1,1,1,1,1],  # malware
    [1,0,1,0,1],  # malware
    [1,1,0,1,1],  # malware
])

y = np.array([0,0,0,1,1,1])

print("Dataset created")
print("Features shape:", X.shape)

# -----------------------------
# STEP 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 3: XGBoost Model
# -----------------------------
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("Training model...")
model.fit(X_train, y_train)

# -----------------------------
# STEP 4: Evaluate
# -----------------------------
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"XGBoost Accuracy: {accuracy:.4f}")

# -----------------------------
# STEP 5: Save Model
# -----------------------------
os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "saved_models/best_model.pkl")

print("Model saved as best_model.pkl")