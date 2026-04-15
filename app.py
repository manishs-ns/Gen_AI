from flask import Flask, render_template, request
import joblib
from pathlib import Path

MODEL_PATH = Path("model.joblib")

def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

model_bundle = load_model()
pipeline = model_bundle["pipeline"]
label_encoder = model_bundle["label_encoder"]

def predict_text(text):
    pred_idx = pipeline.predict([text])[0]
    label = label_encoder.inverse_transform([pred_idx])[0]
    proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        # Map probabilities to labels
        classes = pipeline.classes_
        probs = {label_encoder.inverse_transform([int(c)])[0]: float(p)
                 for c, p in zip(classes, proba)}
    else:
        probs = None
    return label, probs

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", error="Please enter some text")
    label, probs = predict_text(text)
    return render_template("index.html", text=text, label=label, probs=probs)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
