# Model File Placement Guide

## 📁 Model File Location

The **xG model file** must be placed in the **root directory** of this application.

### Expected File Name and Location:
```
xg_interface/
├── xg_model.joblib    ← PLACE YOUR MODEL FILE HERE
├── app.py
├── requirements.txt
├── setup.bat
├── run.bat
└── src/
    └── ...
```

## 🔧 How to Place Your Model

1. **If you have a trained model (.joblib file):**
   - Copy your model file to the `xg_interface/` folder
   - Rename the file to `xg_model.joblib` (exact name required)
   - Ensure the file is at the same level as `app.py`

2. **If you don't have a model yet:**
   - The application will show clear instructions on where to place your model
   - Train your model using your shot data
   - Save it as: `joblib.dump(model, 'xg_model.joblib')`
   - Place it in the root directory

## 📋 Format Model yang Didukung

Model harus berupa **scikit-learn Pipeline**, **LightGBM model**, atau **Calibrated model** yang sudah di-train dengan format:
- **Input**: DataFrame dengan kolom-kolom yang sesuai (lihat `constants.py`)
- **Output**: Probabilitas goal (0-1)
- **Format file**: `.joblib` (menggunakan `joblib.dump()`)

### Model yang didukung:
- Scikit-learn pipelines
- LightGBM classifiers  
- CalibratedClassifierCV dengan LightGBM
- Model lain yang kompatibel dengan joblib

## ✅ Verifikasi

Untuk memastikan model sudah ditempatkan dengan benar:

1. Jalankan aplikasi dengan `run.bat` atau `streamlit run app.py`
2. Jika model ditemukan, aplikasi akan langsung load model Anda
3. Jika model tidak ditemukan, aplikasi akan menampilkan warning dan membuat dummy model

## 🚨 Troubleshooting

**Problem**: Error "ModuleNotFoundError: No module named 'lightgbm'"
- **Solution**: Install LightGBM: `pip install --user lightgbm` or run `setup.bat`

**Problem**: Error "Model file not found"
- **Solution**: Pastikan file bernama `xg_model.joblib` berada di folder root

**Problem**: Error saat loading model
- **Solution**: Pastikan model kompatibel dengan joblib

**Problem**: Prediction error
- **Solution**: Pastikan model di-train dengan fitur yang sama seperti di `constants.py`

## 📝 Catatan Penting

- Model file **TIDAK** di-commit ke git (sudah ada di `.gitignore`)
- Backup model file Anda di tempat yang aman
- Untuk production, gunakan model yang sudah properly trained dan validated
