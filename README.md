# ⚽ xG Prediction Interface v0.0.1

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**A professional web application for Expected Goals (xG) prediction and football shot analysis**

*Transform your football shot data into actionable insights with machine learning*

</div>

---

## 🚀 Features

### 📊 **Batch Prediction**
- Upload CSV files containing shot data for bulk xG predictions
- Process multiple shots simultaneously with detailed results
- Export predictions for further analysis

### 🎯 **Custom Shot Simulation** 
- Interactive shot simulator with adjustable parameters
- Real-time xG calculation based on shot characteristics
- Visual feedback on shot probability

### 📈 **Advanced Visualizations**
- Professional shot maps on football pitch layouts
- Interactive charts and statistics
- Clean, publication-ready visualizations

### 🖼️ **PNG-only Downloads (Streamlit Cloud Ready)**
- All visualization exports are provided as PNG files only
- Plotly static PNG export is attempted first
- Automatic fallback to Matplotlib PNG if Plotly export is not available
- No HTML or other fallback formats presented to users

### 🌍 **Multi-language Support**
- English and Indonesian language options
- Seamless language switching
- Localized user interface

---

## 🛠️ Quick Start

### 💻 **Automated Setup (Recommended)**

**Option 1: Standard Installation**
```bash
setup.bat
```

**Option 2: Virtual Environment (Recommended for developers)**
```bash
setup_venv.bat
```

### 🏃‍♂️ **Running the Application**

**After setup, launch with:**
```bash
run.bat
```

**Or with virtual environment:**
```bash
run_venv.bat
```

**Manual launch:**
```bash
streamlit run app.py
```

---

## 📋 System Requirements

- **Operating System**: Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB free space
- **Internet**: Required for initial setup

---

## 🗂️ Project Architecture

```
xg_interface/
├── 🏠 app.py                       # Main Streamlit application
├── 📋 requirements.txt             # Python dependencies
├── 🔧 setup.bat                    # Automated Windows setup
├── 🚀 run.bat                      # Application launcher
├── 📊 xg_model.joblib              # Trained ML model (place here)
├── 📖 MODEL_PLACEMENT.md           # Model setup guide
├── ⚠️ PATH_WARNING.md              # Troubleshooting guide
└── 📁 src/                         # Source code modules
    ├── 🧠 models/
    │   └── model_manager.py         # ML model management
    ├── 📄 pages/
    │   ├── dataset_prediction.py    # Batch prediction interface
    │   └── custom_shot.py           # Shot simulation interface
    └── 🛠️ utils/
        ├── constants.py             # StatsBomb mappings & constants
        ├── data_processing.py       # Data preprocessing pipeline
        ├── visualization.py         # Chart and plot generation
        ├── visualization_seaborn.py # Alternative plotting helpers
        ├── plotly_export.py         # Robust Plotly PNG exporter
        └── language.py              # Internationalization
```

---

## 📊 Data Format

### CSV Input Requirements

Your dataset must include these **required columns**:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `minute` | int | 0-120 | Match minute |
| `second` | int | 0-59 | Second within minute |
| `play_pattern` | int | StatsBomb ID | Play pattern identifier |
| `position` | int | StatsBomb ID | Player position |
| `shot_technique` | int | StatsBomb ID | Shooting technique |
| `shot_body_part` | int | StatsBomb ID | Body part used |
| `shot_type` | int | StatsBomb ID | Type of shot |
| `shot_open_goal` | bool | 0/1 | Open goal situation |
| `shot_one_on_one` | bool | 0/1 | One-on-one with keeper |
| `shot_aerial_won` | bool | 0/1 | Aerial duel won |
| `under_pressure` | bool | 0/1 | Under defensive pressure |
| `start_x` | float | 0-120 | Pitch X coordinate |
| `start_y` | float | 0-80 | Pitch Y coordinate |
| `type_before` | int | StatsBomb ID | Previous event type |

### 📝 **Sample Data**
```csv
minute,second,play_pattern,position,shot_technique,shot_body_part,shot_type,shot_open_goal,shot_one_on_one,shot_aerial_won,under_pressure,start_x,start_y,type_before
45,30,9,23,93,40,87,0,0,0,1,102.0,36.0,30
67,15,1,23,93,38,87,0,1,0,0,108.5,40.2,42
```

---

## 🎯 Model Information

### **Default Model**
- **Algorithm**: Logistic Regression with advanced preprocessing
- **Features**: 14 engineered features from shot characteristics
- **Preprocessing**: OneHotEncoder for categorical variables
- **Performance**: Optimized for football shot prediction

### **Custom Model Setup**
1. Train your model with the required feature set
2. Save as `xg_model.joblib` using joblib
3. Place in the root directory: `d:\Kuliah\Penelitian\xg_interface\xg_model.joblib`
4. Restart the application

**Note**: If no model is found, the application will display clear instructions on where to place your model file, including the exact path required.

### 📦 Model and Data Licensing
- The included `xg_model.joblib` (if present) is provided for demonstration purposes under the project’s MIT license unless otherwise stated.
- If you train a model using third‑party data (e.g., StatsBomb Open Data), ensure your usage complies with the data provider’s terms. This repository does not grant rights to redistribute third‑party datasets.

---

## 🧪 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | [Streamlit](https://streamlit.io) | Interactive web interface |
| **Data Processing** | [Pandas](https://pandas.pydata.org) | Data manipulation |
| **Machine Learning** | [Scikit-learn](https://scikit-learn.org) | Model training & prediction |
| **Visualization** | [Matplotlib](https://matplotlib.org) + [mplsoccer](https://mplsoccer.readthedocs.io) | Football pitch visualizations |
| **Numerical Computing** | [NumPy](https://numpy.org) | High-performance arrays |
| **Advanced ML** | [LightGBM](https://lightgbm.readthedocs.io) | Gradient boosting |
| **Interactive Plots** | [Plotly](https://plotly.com/python/) | Dynamic visualizations |

---

## 🆘 Troubleshooting

### Common Issues

**🔴 Python Not Found**
```bash
# Solution: Install Python 3.8+ from python.org
# Make sure to check "Add Python to PATH"
```

**🔴 Permission Errors**
```bash
# Run Command Prompt as Administrator, then:
setup.bat
```

**🔴 Package Installation Fails**
```bash
# Use virtual environment:
setup_venv.bat
```

**🔴 Model Loading Error**
- Ensure `xg_model.joblib` is in the root directory
- Check file permissions
- Verify model compatibility

### 📞 **Getting Help**

1. Check `PATH_WARNING.md` for path-related issues
2. Review `MODEL_PLACEMENT.md` for model setup
3. Use virtual environment setup if persistent issues occur
4. Ensure internet connection for package downloads

---

## 📈 Performance Tips

- **Large datasets**: Process in smaller batches for better performance
- **Memory usage**: Close other applications when processing large files
- **Model loading**: Keep model file in root directory for faster loading
- **Visualization**: Limit shot map displays to <1000 points for optimal performance

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style standards
- Testing requirements  
- Documentation updates
- Feature requests

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Additionally:
- Source files include SPDX headers: `SPDX-License-Identifier: MIT`.
- No registration is required to use the MIT license; simply keep the license file and headers.

---

## 🏆 Credits

**Created by:** Fadhil Raihan Akbar  
**Institution:** UIN Syarif Hidayatullah Jakarta — Information Systems  
**Research Purpose:** Undergraduate Thesis — “Application of Light Gradient Boosting Machine (LightGBM) for Expected Goals (xG) Value Prediction in Football Analysis”  
**Contact:**
- GitHub: https://github.com/fadhilra101  
- Instagram: https://www.instagram.com/fadhilra_

**Acknowledgements:**
- Built with Streamlit and the modern Python ML stack
- Visualizations use mplsoccer and Matplotlib; interactive plots via Plotly
- Data format compatible with StatsBomb Open Data (follow their licensing/attribution requirements)

---

<div align="center">

**⚽ Ready to analyze your shots? Get started with `setup.bat`! ⚽**

*Built with ❤️ for the football analytics community*

</div>
