import json
import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from model_utils import RetinaMultiLabelPredictor, LABEL_META

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("RETINA_MODEL_PATH", BASE_DIR / "models" / "retina_multilabel_cnn.pt"))
METRICS_PATH = BASE_DIR / "models" / "training_metrics.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["UPLOAD_EXTENSIONS"] = {".jpg", ".jpeg", ".png", ".webp"}
predictor = RetinaMultiLabelPredictor(MODEL_PATH)


def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/")
def index():
    metrics = load_metrics()
    return render_template(
        "index.html",
        model_ready=predictor.model_ready,
        train_count=metrics.get("train_count", 2748),
        val_count=metrics.get("val_count", 687),
        macro_f1=round(metrics.get("calibrated_macro_f1", 0) * 100, 2),
        micro_f1=round(metrics.get("calibrated_micro_f1", 0) * 100, 2),
    )

@app.get("/api/status")
def status():
    return jsonify({
        "ok": True,
        "model_ready": predictor.model_ready,
        "architecture": predictor.architecture,
        "labels": [{"key": k, "name": v["display"]} for k, v in LABEL_META.items()],
        "thresholds": predictor.thresholds,
    })

@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "File citra tidak ditemukan."}), 400
    image = request.files["image"]
    filename = secure_filename(image.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in app.config["UPLOAD_EXTENSIONS"]:
        return jsonify({"ok": False, "error": "Format harus JPG, JPEG, PNG, atau WEBP."}), 400
    try:
        result = predictor.predict_bytes(image.read(), filename=filename)
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.errorhandler(413)
def too_large(_):
    return jsonify({"ok": False, "error": "Ukuran file melebihi 10 MB."}), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")
