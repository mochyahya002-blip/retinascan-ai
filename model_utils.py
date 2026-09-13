import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageStat

LABEL_META = {
    "opacity": {
        "display": "Opacity",
        "group": "disease",
        "description": "Model menemukan pola yang menyerupai label opacity/kekeruhan pada dataset pelatihan.",
        "next": "Disarankan pemeriksaan mata langsung untuk memastikan lokasi dan penyebab kekeruhan."
    },
    "diabetic retinopathy": {
        "display": "Retinopati Diabetik",
        "group": "disease",
        "description": "Model menemukan pola citra yang menyerupai retinopati diabetik pada dataset.",
        "next": "Bila pasien memiliki diabetes atau keluhan penglihatan, pertimbangkan evaluasi dokter mata dan pemeriksaan retina."
    },
    "glaucoma": {
        "display": "Glaukoma",
        "group": "disease",
        "description": "Model menemukan pola yang pada dataset dikaitkan dengan glaukoma.",
        "next": "Glaukoma tidak dapat dipastikan dari model ini saja. Pemeriksaan tekanan intraokular, saraf optik, dan lapang pandang tetap diperlukan."
    },
    "macular edema": {
        "display": "Edema Makula",
        "group": "disease",
        "description": "Model menemukan pola yang menyerupai edema pada area makula.",
        "next": "Pertimbangkan evaluasi dokter mata; pemeriksaan tambahan seperti OCT dapat diperlukan sesuai penilaian klinis."
    },
    "macular degeneration": {
        "display": "Degenerasi Makula",
        "group": "disease",
        "description": "Model menemukan pola yang menyerupai degenerasi makula pada dataset pelatihan.",
        "next": "Disarankan pemeriksaan retina/makula oleh dokter mata untuk konfirmasi dan penilaian derajat kelainan."
    },
    "retinal vascular occlusion": {
        "display": "Oklusi Vaskular Retina",
        "group": "disease",
        "description": "Model menemukan pola yang menyerupai oklusi pembuluh darah retina.",
        "next": "Keluhan penurunan penglihatan mendadak perlu dinilai segera oleh tenaga medis atau dokter mata."
    },
    "normal": {
        "display": "Normal",
        "group": "normal",
        "description": "Model lebih mendukung label normal berdasarkan pola yang dipelajari dari dataset.",
        "next": "Hasil normal dari AI tidak menyingkirkan seluruh penyakit mata. Tetap periksakan mata bila ada keluhan atau faktor risiko."
    },
}

class SmallRetinaCNN:
    """Factory wrapper agar torch hanya diimpor saat diperlukan."""
    @staticmethod
    def build(num_classes=7):
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                def block(i, o):
                    return nn.Sequential(
                        nn.Conv2d(i, o, 3, padding=1, bias=False),
                        nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2)
                    )
                self.features = nn.Sequential(
                    block(3, 16), block(16, 32), block(32, 64), block(64, 96)
                )
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Sequential(
                    nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, 64),
                    nn.ReLU(inplace=True), nn.Linear(64, num_classes)
                )
            def forward(self, x):
                return self.head(self.pool(self.features(x)))
        return Net()

class RetinaMultiLabelPredictor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.labels = list(LABEL_META.keys())
        self.image_size = 96
        self.thresholds = {name: 0.5 for name in self.labels}
        self.architecture = "SmallRetinaCNN-v1"
        self.model_ready = False
        self.load()

    def load(self):
        if not self.model_path.exists():
            return
        import torch
        try:
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(self.model_path, map_location="cpu")
        self.labels = ckpt.get("labels", self.labels)
        self.image_size = int(ckpt.get("image_size", 96))
        self.architecture = ckpt.get("architecture", self.architecture)
        if "thresholds" in ckpt:
            self.thresholds.update({k: float(v) for k, v in ckpt["thresholds"].items()})
        elif "threshold" in ckpt:
            self.thresholds = {name: float(ckpt["threshold"]) for name in self.labels}
        self.model = SmallRetinaCNN.build(len(self.labels))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.model_ready = True

    def _open_image(self, raw):
        if not raw:
            raise ValueError("File citra kosong.")
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise ValueError("File tidak dapat dibaca sebagai citra.") from exc

    def _quality(self, image):
        warnings = []
        w, h = image.size
        gray = image.convert("L")
        mean = float(ImageStat.Stat(gray).mean[0])
        if min(w, h) < 128:
            warnings.append("Resolusi citra rendah; gunakan gambar yang lebih besar bila tersedia.")
        if mean < 35:
            warnings.append("Citra sangat gelap sehingga prediksi dapat kurang stabil.")
        elif mean > 225:
            warnings.append("Citra sangat terang sehingga detail retina dapat berkurang.")
        return {"width": w, "height": h, "mean_brightness": round(mean, 1), "warnings": warnings}

    def _tensor(self, image):
        import torch
        img = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return torch.from_numpy(arr)

    def predict_bytes(self, raw, filename="image"):
        if not self.model_ready:
            raise RuntimeError("Model belum tersedia di folder models/.")
        import torch
        image = self._open_image(raw)
        quality = self._quality(image)
        x = self._tensor(image)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(x))[0].cpu().numpy().astype(float)

        by_label = {name: float(p) for name, p in zip(self.labels, probs)}
        disease_labels = [n for n in self.labels if n != "normal"]
        positive_diseases = [n for n in disease_labels if by_label[n] >= self.thresholds[n]]
        normal_positive = by_label.get("normal", 0.0) >= self.thresholds.get("normal", 0.5)

        if positive_diseases:
            detected = positive_diseases
            state = "perlu-evaluasi"
            summary = f"{len(detected)} label penyakit melewati threshold model."
        elif normal_positive:
            detected = ["normal"]
            state = "normal"
            summary = "Label normal melewati threshold dan tidak ada label penyakit yang melewati threshold."
        else:
            detected = []
            state = "tidak-pasti"
            summary = "Tidak ada label yang melewati threshold; hasil perlu dianggap tidak pasti."

        ranked = sorted(self.labels, key=lambda n: by_label[n], reverse=True)
        details = []
        for name in detected:
            meta = LABEL_META[name]
            details.append({
                "key": name,
                "name": meta["display"],
                "probability": round(by_label[name] * 100, 2),
                "threshold": round(self.thresholds[name] * 100, 1),
                "description": meta["description"],
                "next_step": meta["next"],
            })

        return {
            "mode": "trained-model",
            "filename": filename,
            "state": state,
            "summary": summary,
            "detected": details,
            "top_label": LABEL_META[ranked[0]]["display"],
            "top_probability": round(by_label[ranked[0]] * 100, 2),
            "probabilities": [
                {
                    "key": name,
                    "name": LABEL_META[name]["display"],
                    "value": round(by_label[name] * 100, 2),
                    "threshold": round(self.thresholds[name] * 100, 1),
                    "positive": name in detected,
                }
                for name in self.labels
            ],
            "quality": quality,
            "model": {
                "architecture": self.architecture,
                "input_size": f"{self.image_size}×{self.image_size}",
                "activation": "Sigmoid",
                "classes": len(self.labels),
            },
            "disclaimer": "Prototype AI untuk skrining berbasis dataset kompetisi. Bukan alat diagnosis medis dan belum tervalidasi untuk penggunaan klinis."
        }
