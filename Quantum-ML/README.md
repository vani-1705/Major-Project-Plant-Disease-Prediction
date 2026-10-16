# ⚛️ Quantum ML - Plant Disease Prediction

A web app that finds potato leaf diseases from a leaf image, text, or voice using quantum models (QSVM, VQC, QNN) and compares them with classical models (SVM, Random Forest, MLP). Part of the major project "Quantum ML vs Classical ML on Plant Disease Prediction".

🔗 Live demo: https://major-project-plant-disease-prediction-4l0d.onrender.com

## Note: quantum models run on a simulator (Qiskit), not a real quantum computer.

## ✨ Features

- 📷 Image: upload a leaf photo and choose any of the 6 models
- 💬 Text and 🎤 voice in English or Telugu
- 🌱 Health tips for each disease
- 📊 Model Graphs page to compare all models
- 🕒 History and 👤 login

## 🌱 Plant

Potato only: Early blight, Late blight and Healthy.

## 🔀 Different from Classical-ML

- Only potato (3 classes), because quantum simulation is slow
- Images reduced to 4 features, one qubit for each feature
- 120 training images and the same 60 test images for all 6 models, so the comparison is fair
- Quantum models are saved without pickle (see below)
- index.html is inside the templates folder

## ⚛️ Quantum models

- QSVM: a quantum kernel finds how similar two leaves are, then an SVM classifies
- VQC: features go into 4 qubits and a trainable circuit learns the classes
- QNN: same circuit as VQC, built as a quantum neural network

## 🧩 Pickle-free saving

Quantum models saved with pickle load only on the same Python version, which caused errors on Render. So we save only simple files (weights and arrays), and qml_runtime.py rebuilds the circuits and predicts. The website needs only qiskit, numpy, scikit-learn and joblib.

## 🛠️ Tech stack

Python, Flask, Qiskit, scikit-learn, OpenCV, SQLite, HTML, CSS, JavaScript, Chart.js, Gunicorn

## 📁 Structure

```
Quantum-ML/
├── app.py
├── qml_runtime.py
├── gunicorn.conf.py
├── requirements.txt
├── .python-version
├── Quantum_ML_PlantDisease.ipynb
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   ├── script.js
│   ├── images/
│   │   ├── Home.jpg
│   │   ├── About.jpg
│   │   ├── Model_Graphs.jpg
│   │   ├── History.jpg
│   │   └── User.jpg
│   └── samples/
│       ├── Potato___Early_blight_1.jpg  (3 images)
│       ├── Potato___healthy_1.jpg       (3 images)
│       └── Potato___Late_blight_1.jpg   (3 images)
│
├── data/
│   ├── tips.json
│   ├── keywords.json
│   ├── model_results.json
│   └── comparison_results.csv
│
└── models/
    ├── classes.json
    ├── preprocess.pkl
    ├── svm.pkl
    ├── rf.pkl
    ├── mlp.pkl
    ├── qsvm_svc.pkl
    ├── qsvm_train_X.npy
    ├── vqc_weights.npy
    └── qnn_weights.npy
```

## ▶️ Run locally

```
cd Quantum-ML
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open http://127.0.0.1:5000 (the first quantum prediction is slow).

## ☁️ Deploy on Render

Root Directory: Quantum-ML
Build: `pip install -r requirements.txt`
Start: `gunicorn app:app -c gunicorn.conf.py`
Set SECRET_KEY and PYTHON_VERSION (3.11.9) as environment variables. On the 512 MB free plan, add WARMUP = 0 if memory runs out.

## ⚠️ Limitations

- Only potato leaves
- Simulator is slow and uses only 4 qubits and 120 training images
- Only English and Telugu for text and voice
- For guidance only, ask an agriculture expert before using pesticides

## 🔮 Future work

Real quantum hardware, more qubits, more data and more plants.

## 👩‍💻 Author

vani-1705 (https://github.com/vani-1705)
