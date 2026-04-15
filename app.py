import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

model = joblib.load("churn_model.joblib")
data = load_breast_cancer()
feature_names = list(data.feature_names)

@app.route("/")
def home():
    return jsonify({
        "message": "Customer Churn Prediction API",
        "required_features": feature_names
    })

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json

    df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0].max()

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
