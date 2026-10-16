# 🌿 Classical ML - Plant Disease Prediction

A web app that finds plant diseases from a leaf image, text, or voice and gives simple health tips. Part of the major project "Quantum ML vs Classical ML on Plant Disease Prediction".

🔗 Live demo: https://major-project-plant-disease-prediction.onrender.com

## ✨ Features

- 📷 Image: upload a leaf photo, get the top 3 likely diseases
- 💬 Text: describe the problem in English or Telugu
- 🎤 Voice: speak in English or Telugu (Chrome or Edge)
- 🌱 Health tips for each disease
- 📊 Model Performance page with accuracy, precision, recall, F1 and confusion matrix
- 🕒 History and 👤 login

## 🌱 Plants

Pepper bell, Potato and Tomato (15 classes, PlantVillage dataset). Other plants may give wrong answers.

## 🧠 How it works

1. Remove corrupted and duplicate images, resize to 64x64
2. Extract HOG and colour histogram features
3. Scale and reduce with PCA
4. Train SVM, Random Forest and MLP
5. Text and voice use a keyword matcher

## 🛠️ Tech stack

Python, Flask, scikit-learn, OpenCV, SQLite, HTML, CSS, JavaScript, Chart.js

## 📁 Structure

```
Classical-ML/
├── app.py                          # Flask backend (API, login, history)
├── requirements.txt
├── .gitignore
├── Classical_ML_PlantDisease.ipynb # training notebook (Google Colab)
│
├── static/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   ├── tips.json
│   ├── images/
│   │   ├── home.jpg, about.jpg, model_performance.jpg
│   │   ├── history.jpg, user.jpg, leaf-bg.jpg
│   │   └── disease-image1.jpg, disease-image2.jpg, disease-image3.jpg
│   └── samples/
│       ├── Potato___Early_blight_1.jpg  (3 images)
│       ├── Potato___healthy_1.jpg       (3 images)
│       └── Potato___Late_blight_1.jpg   (3 images)
│
├── data/
│   ├── keywords.json
│   └── model_performance.json
│
└── models/
    ├── app_SVM.pkl
    ├── app_RandomForest.pkl
    ├── app_MLP.pkl
    ├── app_classes.json
    ├── app_pca.pkl
    └── app_scaler.pkl
```

## ▶️ Run locally

```
cd Classical-ML
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open http://127.0.0.1:5000

## ☁️ Deploy on Render

Build: `pip install -r requirements.txt`
Start: `gunicorn app:app`
Set SECRET_KEY as an environment variable.

## ⚠️ Limitations

- Only Pepper bell, Potato and Tomato
- Only English and Telugu for text and voice
- For guidance only, ask an agriculture expert before using pesticides
- On the free plan the app can take a minute to wake up

## 🔮 Future work

Quantum models (see the Quantum-ML folder), more plants and languages.

## 👩‍💻 Author

vani-1705 (https://github.com/vani-1705)
