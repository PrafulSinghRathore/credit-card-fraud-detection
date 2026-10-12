# 💳 Credit Card Fraud Detection

A machine learning-powered system for detecting fraudulent credit card transactions in real-time. Built with Scikit-learn, XGBoost, and Flask.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey.svg)

---

## 📋 Project Overview

Credit card fraud is a significant financial threat. This project uses machine learning to identify fraudulent transactions with high recall, minimizing missed fraud cases.

### Key Features
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
- **Imbalanced Data Handling**: SMOTE oversampling technique
- **Real-time Prediction**: Flask-based web interface with animated gauge
- **Batch CSV Dataset Screening**: Drag-and-drop any CSV dataset to screen up to 50,000 transactions at once
- **Comprehensive EDA**: 9 high-resolution visualization charts
- **Model Comparison**: Detailed performance metrics, ROC curves & PR curves

---

## 🗂️ Project Structure

```
credit_card_fraud_detection/
├── train_model.py          # Model training pipeline
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── data/
│   └── creditcard.csv      # Dataset (download separately)
├── models/                 # Saved trained models
│   ├── best_model.pkl
│   ├── scaler_amount.pkl
│   ├── scaler_time.pkl
│   └── model_info.json
└── static/
    ├── index.html           # Frontend UI
    ├── style.css            # Stylesheet
    ├── script.js            # Frontend logic
    └── charts/              # Generated EDA & evaluation charts
```

---

## 🚀 Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download the **Credit Card Fraud Detection** dataset from Kaggle:
- **URL**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Place `creditcard.csv` inside the `data/` folder

### 3. Train the Model
```bash
python train_model.py
```
This will:
- Load and analyze the dataset (EDA)
- Preprocess data (scaling, SMOTE)
- Train 3 models (Logistic Regression, Random Forest, XGBoost)
- Generate evaluation charts
- Save the best model

### 4. Run the Web App
```bash
python app.py
```
Open your browser at: **http://localhost:5000**

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transactions**: 284,807
- **Fraudulent**: 492 (0.172%)
- **Features**: V1-V28 (PCA transformed), Time, Amount
- **Target**: Class (0 = Normal, 1 = Fraud)

---

## 🤖 Models Used

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear classifier, baseline model |
| **Random Forest** | Ensemble of decision trees with bagging |
| **XGBoost** | Gradient boosting with regularization |

### Handling Class Imbalance
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **Class weights** in Random Forest
- **scale_pos_weight** in XGBoost

---

## 📈 Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of predicted frauds, how many are actually fraud
- **Recall**: Of actual frauds, how many were detected (most important!)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Precision-Recall AUC**: Better metric for imbalanced datasets

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Scikit-learn** — ML models & preprocessing
- **XGBoost** — Gradient boosting
- **Imbalanced-learn** — SMOTE
- **Pandas & NumPy** — Data manipulation
- **Matplotlib & Seaborn** — Visualization
- **Flask** — Web framework
- **HTML/CSS/JS** — Frontend UI

---

## 👨‍💻 Author

**B.Tech CSE — 5th Semester**
Machine Learning Lab Project

---

## 📄 License

This project is for educational purposes only.
