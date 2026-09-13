# RetinaScan AI — CNN Multi-label

Website screening citra retina menggunakan model CNN **7 output sigmoid**. Model prototype yang disertakan dilatih dari dataset yang memiliki 3.435 citra training dan label:

`opacity`, `diabetic retinopathy`, `glaucoma`, `macular edema`, `macular degeneration`, `retinal vascular occlusion`, `normal`.

Satu citra dapat memiliki beberapa label penyakit. Pada dataset, label `normal` tidak muncul bersamaan dengan label penyakit.

## Menjalankan website

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:5000`.

## Model yang sudah disertakan

`models/retina_multilabel_cnn.pt` adalah model prototype hasil training pada dataset yang diberikan. Preprocessing: resize 96×96 RGB dan normalisasi ke rentang -1..1. Output: 7 logits → sigmoid.

Threshold dikalibrasi per kelas pada validation split agar tidak memaksakan threshold 0.5 yang sama untuk semua label. File `models/training_metrics.json` menyimpan metrik eksperimen dan threshold tersebut.

> **Penting:** performa validation prototype ini masih terbatas dan belum divalidasi secara klinis. Website tidak boleh diposisikan sebagai alat diagnosis atau pengganti dokter.

## Training ulang

1. Ekstrak `train.zip` sehingga gambar berada di `data/train/`.
2. Salin `train.csv` ke `data/train.csv`.
3. Instal dependency training dan jalankan:

```bash
pip install -r requirements-train.txt
python scripts/train.py
```

Training produksi sebaiknya menambahkan augmentasi yang lebih kuat, stratifikasi multi-label, transfer learning/pretraining, cross-validation, tuning threshold pada validation set terpisah, dan external validation.

## Test set / submission

Ekstrak `test.zip` ke `data/test/`, letakkan `sample_submission.csv` di `data/`, lalu:

```bash
python scripts/predict_test.py
```

Hasil disimpan sebagai `prediction_submission.csv`.

## Deploy publik paling mudah: GitHub + Render

File deploy untuk Render sudah disertakan:

- `requirements-render.txt` — dependency server, termasuk PyTorch CPU dan Gunicorn.
- `.python-version` — memakai Python 3.12 untuk kompatibilitas dependency.
- `render.yaml` — konfigurasi Render Free Web Service.

Jika membuat Web Service secara manual di Render, gunakan:

- Build Command: `pip install -r requirements-render.txt`
- Start Command: `gunicorn --workers 1 --threads 2 --timeout 120 app:app`
- Health Check Path: `/api/status`

Model `models/retina_multilabel_cnn.pt` harus ikut berada di repository GitHub karena dipakai saat startup.
