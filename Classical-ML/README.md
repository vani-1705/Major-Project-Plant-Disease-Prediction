# 🌿⚛️ Major Project: Quantum ML vs Classical ML on Plant Disease Prediction

This project checks whether **quantum machine learning** can detect plant leaf diseases as well as **classical machine learning**. We built two working websites, one for each, using the same PlantVillage dataset.

## 🌐 Live demos

- 🌿 Classical ML: https://major-project-plant-disease-prediction-4l0d.onrender.com
- ⚛️ Quantum ML: https://YOUR-QUANTUM-RENDER-LINK.onrender.com

## 🎯 The idea

Plant diseases reduce crop yield and farmer income. A farmer should be able to take a leaf photo, or just describe the problem in Telugu or English, and quickly know the disease and what to do. We built this with two approaches and compared them.

## 🌿 Project 1: Classical-ML

Uses normal machine learning models.

- 🤖 Models: SVM, Random Forest, MLP
- 🌱 Plants: Pepper bell, Potato and Tomato (15 classes)
- 📷 Input: leaf image, text, or voice (English and Telugu)
- 📊 Pages: Home, About, Model Performance, History, User login
- 🧠 How: image is cleaned and resized, HOG and colour features are taken, scaled, reduced with PCA, then given to the models

## ⚛️ Project 2: Quantum-ML

Uses quantum models running on a simulator (Qiskit), along with the classical models for comparison.

- ⚛️ Quantum models: QSVM, VQC, QNN
- 🤖 Classical models: SVM, Random Forest, MLP (same data, for fair comparison)
- 🌱 Plant: Potato only (Early blight, Late blight, Healthy)
- 📷 Input: leaf image, text, or voice (English and Telugu), and the user can choose any of the 6 models
- 📊 Pages: Home, About, Model Graphs, History, User login
- 🧠 How: same features as above, but reduced to only 4 values with PCA and scaled to 0 to pi, so 4 qubits can use them

## 🔀 Main differences

| | 🌿 Classical-ML | ⚛️ Quantum-ML |
|---|---|---|
| Models | 3 classical | 3 quantum + 3 classical |
| Plants | Pepper, Potato, Tomato | Potato only |
| Classes | 15 | 3 |
| Features per image | Many (after PCA) | 4 (one qubit each) |
| Training images | Larger set | 120 images |
| Test images | Same split | Same 60 test images |
| Speed | Fast | Slow (simulator) |
| Model saving | Normal `.pkl` files | Pickle-free (`.pkl` + `.npy` + `qml_runtime.py`) |
| HTML location | `static/index.html` | `templates/index.html` |

## ❓ Why is Quantum-ML smaller?

A quantum simulator needs one qubit for each feature, and its time grows very fast when qubits and data increase. So we used 4 features, 120 training images and only potato. The classical models in the Quantum-ML project use the same data, so the comparison between quantum and classical is fair.

## 🧩 What we did differently in Quantum-ML

Quantum models saved with pickle load only on the same Python version, which gave errors on Render. So we save only simple files: the trained weights and arrays. A small file, `qml_runtime.py`, rebuilds the same circuits and makes the prediction. Because of this, the website runs on any Python version.

## 📁 Repository structure

```
Major-Project-Plant-Disease-Prediction/
├── README.md
├── Classical-ML/
│   ├── app.py
│   ├── requirements.txt
│   ├── Classical_ML_PlantDisease.ipynb
│   ├── static/        (index.html, style.css, script.js, tips.json, images, samples)
│   ├── data/          (keywords.json, model_performance.json)
│   └── models/        (SVM, Random Forest, MLP, PCA, scaler, classes)
│
└── Quantum-ML/
    ├── app.py
    ├── qml_runtime.py
    ├── gunicorn.conf.py
    ├── requirements.txt
    ├── .python-version
    ├── Quantum_ML_PlantDisease.ipynb
    ├── templates/     (index.html)
    ├── static/        (style.css, script.js, images, samples)
    ├── data/          (tips.json, keywords.json, model_results.json, comparison_results.csv)
    └── models/        (classical and quantum model files)
```

Each folder has its own README with full details.

## ▶️ Run locally

Run any one project, the steps are the same:

```
cd Classical-ML        # or: cd Quantum-ML
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 app.py
```

Open http://127.0.0.1:5000 in your browser.

## ☁️ Deploy on Render

Both projects are deployed as separate Render web services from this same repository.

- Root Directory: `Classical-ML` or `Quantum-ML`
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app` (Classical) or `gunicorn app:app -c gunicorn.conf.py` (Quantum)
- Environment variable: `SECRET_KEY` = any long random text

## 🛠️ Tech stack

Python, Flask, scikit-learn, Qiskit, OpenCV, scikit-image, SQLite, HTML, CSS, JavaScript, Chart.js, Web Speech API, Gunicorn

## 📊 Dataset

PlantVillage (Kaggle): https://www.kaggle.com/datasets/emmarex/plantdisease

## ⚠️ Limitations

- Classical works for Pepper bell, Potato and Tomato. Quantum works for Potato only
- Quantum models run on a simulator, not a real quantum computer, so they are slow
- Text and voice support only English and Telugu
- Results are for guidance. Ask an agriculture expert before using pesticides
- On the free hosting plan, the app can take a minute to wake up and the database can reset

## 🔮 Future work

- Run quantum models on real quantum hardware
- More qubits, more data and more plants
- More Indian languages and real field images

## 👩‍💻 Author

vani-1705 (https://github.com/vani-1705)
