# Fraud Detection Project - README

## Overview
This project tackles fraud detection on two datasets:  
- E-Commerce Fraud Dataset (`Fraud_Data.csv`)  
- Credit Card Transaction Dataset (`creditcard.csv`)  

It includes three main tasks:  
1. **Data Preparation & Feature Engineering**  
2. **Model Building & Evaluation** using Logistic Regression and Random Forest with SMOTE for imbalance  
3. **Model Explainability** using SHAP for feature importance interpretation  

---

## Environment Setup

### 1. Python Version
- Use Python 3.8 or later.

### 2. Create & Activate Virtual Environment (Recommended)
```bash
python -m venv fraud_env
# Windows
.\fraud_env\Scripts\activate
# Linux/Mac
source fraud_env/bin/activate
###Install Dependencies

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap
If SHAP gives errors, upgrade these packages:


pip install --upgrade scipy numba