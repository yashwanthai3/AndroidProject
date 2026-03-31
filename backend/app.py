from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_image   # ✅ correct import
import os

app = Flask(__name__)
CORS(app)


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Create temp folder if not exists
    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)

    path = os.path.join(temp_folder, file.filename)
    file.save(path)

    try:
        prediction, confidence = predict_image(path)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
