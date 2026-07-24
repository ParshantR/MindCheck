# 🧠 MindCheck

An AI-powered Mental Health Assessment web application based on the **DASS-21 (Depression Anxiety Stress Scale)** questionnaire. The application predicts the severity levels of **Stress**, **Anxiety**, and **Depression** using a Machine Learning Soft Voting Ensemble model.

---

## 📌 Overview

MindCheck is designed to provide a fast and user-friendly mental health self-assessment experience. Users answer the standard DASS-21 questionnaire along with a few demographic questions, and the application predicts their mental health severity levels using trained machine learning models.

**Disclaimer:** This application is developed for educational and research purposes only. It is **not** a substitute for professional medical diagnosis or treatment.

---

## ✨ Features

- 📋 DASS-21 questionnaire based assessment
- 👤 Demographic information collection
- 🤖 AI-powered prediction using Soft Voting Ensemble
- 📊 Individual prediction for:
  - Stress
  - Anxiety
  - Depression
- 📈 Prediction confidence charts
- 🎯 Overall mental health risk score
- 💻 Responsive and modern web interface
- ⚡ Instant prediction results
- 🔒 No user data is permanently stored

---

## 🧠 Machine Learning Model

The application uses a **Soft Voting Ensemble** consisting of:

- HistGradientBoosting Classifier
- Random Forest Classifier
- Dropout Multi-Layer Perceptron (MLP)

Each target (Stress, Anxiety and Depression) has its own independently trained ensemble model.

---

## 📊 Model Performance

| Target | Accuracy | Macro F1 | Cohen's Kappa |
|---------|---------:|---------:|--------------:|
| Stress | **79.9%** | **0.714** | **0.715** |
| Anxiety | **74.9%** | **0.669** | **0.657** |
| Depression | **80.4%** | **0.760** | **0.742** |

These performance metrics were obtained on a held-out test dataset.

---

## 🛠 Tech Stack

### Backend

- Python
- Flask
- Flask-CORS
- Scikit-learn
- NumPy
- Joblib

### Frontend

- HTML5
- CSS3
- JavaScript
- Chart.js

### Machine Learning

- HistGradientBoosting
- Random Forest
- Multi-Layer Perceptron (MLP)
- Soft Voting Ensemble

---

## 📁 Project Structure

```
MindCheck/
│
├── backend/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── templates/
│   └── static/
│       ├── css/
│       ├── js/
│       └── images/
│
├── models/
│   ├── stress_model.pkl
│   ├── anxiety_model.pkl
│   ├── depression_model.pkl
│   ├── scaler.pkl
│   └── metadata.json
│
└── README.md
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/ParshantR/MindCheck.git
```

### Move into the project

```bash
cd MindCheck
```

### Create virtual environment

```bash
python -m venv venv
```

### Activate virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r backend/requirements.txt
```

### Start Backend

```bash
cd backend
python app.py
```

### Start Frontend

Open another terminal:

```bash
cd frontend
python -m http.server 8080
```

Visit:

```
http://localhost:8080/templates/index.html
```

---

## 📈 Future Improvements

- User authentication
- Assessment history
- PDF report generation
- Doctor dashboard
- Email report delivery
- Cloud database integration
- Improved mobile responsiveness
- Explainable AI visualizations

---

## 👨‍💻 Developer

**Parshant Ratawal**

M.Sc. Data Science  
VIT Vellore

GitHub: https://github.com/ParshantR

---

## 📜 License

This project is developed for academic and educational purposes.
