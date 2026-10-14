# 🌿 Plant Disease Detection (Classical ML)

A web application that detects plant diseases from a **leaf image, text, or voice** and gives simple **health tips**. Part of the major project **Quantum ML vs Classical ML on Plant Disease Prediction**.

🔗 **Live demo:** https://major-project-plant-disease-prediction.onrender.com/

## ✨ Features

- **Image source:** upload a leaf photo, get the top 3 likely diseases with confidence
- **Text source:** describe the problem in English or Telugu
- **Voice source:** speak in English or Telugu (Chrome or Edge), speech is converted to text
- **Health tips** for the predicted disease, with a Listen button
- **Model Performance page:** accuracy, precision, recall, F1, confusion matrix, learning curve
- **History page:** past predictions by Image, Text and Voice source (after login)
- **User section:** sign up, login, logout

## 🌱 Supported plants and diseases

Pepper bell, Potato and Tomato (15 classes from the PlantVillage dataset). Other plants may give wrong answers, so low-confidence results show "Not sure".

## 🧠 How it works

1. Images are cleaned (corrupted and duplicate images removed) and resized to 64x64
2. Features are extracted: **HOG** (shape) and **colour histogram**
3. Features are scaled (StandardScaler) and reduced with **PCA**
4. Classical models are trained: **SVM, Random Forest, MLP**
5. The website uses the trained model to predict, and a keyword matcher for text and voice

## 🛠️ Tech stack

Python, Flask, scikit-learn, OpenCV, scikit-image, SQLite, HTML, CSS, JavaScript, Chart.js, Web Speech API

## 📁 Project structure

```
Plant-Disease-Prediction/
└── Classical-ML/
    ├── app.py                 # Flask backend (API, login, history)
    ├── requirements.txt
    ├── data/                  # keywords.json, model_performance.json
    ├── models/                # trained model files (.pkl) and classes
    ├── static/                # frontend (index.html, style.css, script.js, tips.json, images, samples)
    └── Classical_ML_PlantDisease.ipynb   # training notebook (Google Colab)
```

## ▶️ Run locally

```bash
cd Classical-ML
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 app.py
```

Open http://127.0.0.1:5000 in your browser.

## 🔌 API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/predict` | Predict disease from an image |
| POST | `/api/text` | Predict disease from text or voice text |
| POST | `/api/register`, `/api/login`, `/api/logout` | User account |
| GET | `/api/history` | Past predictions of the logged-in user |
| GET | `/api/performance` | Model performance data |

## 📊 Dataset

[PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) (Kaggle).

## ⚠️ Limitations

- Works only for Pepper bell, Potato and Tomato
- Text and voice support English and Telugu only
- Predictions are for guidance. Consult an agriculture expert before using pesticides
- On the free hosting plan, the app may take a minute to wake up and the database can reset

## 🔮 Future work

- Quantum models (QSVM, VQC) and the Classical vs Quantum comparison
- More plants and more Indian languages
- Real field images for better accuracy

## 👩‍💻 Author

[vani-1705](https://github.com/vani-1705)
