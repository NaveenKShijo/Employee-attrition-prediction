# 📈 Employee Activation Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Classification-green)

An end-to-end Machine Learning project designed to predict and analyze employee activation rates. This repository includes the full research workflow in Jupyter Notebooks and a functional web interface built with Streamlit.

---

## 📋 Project Overview
The goal of this project is to provide a data-driven approach to understanding employee engagement. By predicting the "activation rate," organizations can identify which segments of their workforce are likely to be active or disengaged, allowing for proactive HR interventions.



### Key Features
* **Comprehensive Data Analysis:** Exploratory Data Analysis (EDA) and preprocessing steps handled in Python.
* **Multi-Model Benchmarking:** Comparative analysis of 7 different classification algorithms.
* **Interactive UI:** A user-friendly dashboard where users can input parameters and get instant predictions.
* **Model Persistence:** Optimized models saved as `.pkl` files for high-speed inference.

---

## 🛠️ Technologies & Tools
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
* **Deployment:** Streamlit
* **Environment:** Jupyter Notebook / VS Code

---

## 🧪 Machine Learning Workflow

### 1. Models Evaluated
I implemented and compared several algorithms to determine which provided the most reliable predictions:
* **Logistic Regression** (Selected Model)
* **Support Vector Machine (SVM)**
* **K-Nearest Neighbors (KNN)**
* **Decision Tree Classifier**
* **Random Forest Classifier**
* **Bagging Models**
* **XGBoost**

### 2. Evaluation Metrics
Each model was rigorously tested using the following metrics to ensure a balanced view of performance:
* **Precision:** To minimize false positives.
* **Recall:** To ensure we don't miss "active" cases.
* **F1-Score:** To find the harmonic mean between the two.

### 3. The Final Choice: Logistic Regression
After evaluating the performance across all metrics, **Logistic Regression** was chosen for the final production environment. It provided a high level of accuracy while maintaining **interpretability** and **low computational overhead**, making it ideal for the Streamlit UI.

---

## 📂 Repository Structure
```bash
├── data/                  # Datasets used for training
├── models/                # Saved .pickle files (Logistic Regression, etc.)
├── activation_analysis.ipynb  # Jupyter notebook containing EDA & Model training
├── app.py                 # Streamlit application source code
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
