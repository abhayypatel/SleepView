# 💤 SleepView – Predict Sleep Disorders with AI

SleepView is a full-stack web app that uses machine learning to predict sleep disorders like insomnia or sleep apnea based on your health and lifestyle data. It’s fast, lightweight, and privacy-friendly — no data is stored.

[![SleepView](https://img.shields.io/badge/SleepView-AI%20Sleep%20Prediction-6366f1?style=for-the-badge)](https://github.com/abhayypatel/SleepView)

**LIVE DEMO**: [Click Here](https://sleep-view.vercel.app/)

---

## 🔍 Features

- Predicts **Sleep Apnea**, **Insomnia**, or **No Disorder**
- Works in real-time with confidence scores
- Clean React frontend + Flask backend
- Models: **XGBoost**, **Random Forest**, **Logistic Regression**
- Dataset: [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

---

## 🧠 Model Accuracy

| Model               | Accuracy |
|--------------------|----------|
| XGBoost            | 96.8%    |
| Random Forest      | 95.2%    |
| Logistic Regression| 92.5%    |

---

## ⚙️ Stack

**Frontend:** React, Axios, React Router  
**Backend:** Flask, Scikit-learn, XGBoost, Pandas  
**ML Tools:** Jupyter, Pickle, NumPy

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/abhayypatel/SleepView
cd SleepView
```

### 2. Train the model

```bash
pip install -r backend/requirements.txt
jupyter notebook sleepDetectionModel.ipynb
# Run all cells to generate model.pkl
```

### 3. Start the backend

```bash
cd backend
python app.py
# Runs at http://localhost:5000
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm start
# Opens http://localhost:3000
```
---

## 🌐 Example API Call

### POST /predict
```
{
  "gender": "Male",
  "age": 30,
  "occupation": "Software Engineer",
  "sleep_duration": 7,
  "quality_of_sleep": 8,
  "physical_activity_level": 60,
  "stress_level": 4,
  "bmi_category": "Normal",
  "blood_pressure_systolic": 120,
  "blood_pressure_diastolic": 80,
  "heart_rate": 70,
  "daily_steps": 8000
}
```

### Response
```
{
  "prediction": "None",
  "confidence": 0.95,
  "probabilities": {
    "None": 0.95,
    "Sleep Apnea": 0.03,
    "Insomnia": 0.02
  },
  "status": "success"
}
```
---

**Built with ❤️ by Abhay for better sleep health** 
