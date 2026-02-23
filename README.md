# ML Based Blood Disease Diagnosis System

## Project Overview

This project presents a Machine Learning based system for the diagnosis of blood-related disorders using laboratory test data. The system analyzes Complete Blood Count (CBC), High Performance Liquid Chromatography (HPLC), and Scatter parameters to classify blood diseases using supervised learning algorithms.

The objective of this project is to assist in early detection, improve diagnostic efficiency, and support medical decision-making using data-driven techniques.

---

## Problem Statement

Manual diagnosis of hematological disorders requires expert interpretation and can be time consuming. This project aims to:

- Automate disease classification
- Improve prediction accuracy
- Reduce diagnosis time
- Provide decision support using ML models

---

## Dataset Description

The dataset consists of three main components:

### 1. CBC Dataset
Contains hematological parameters such as:
- Hemoglobin (Hb)
- RBC count
- WBC count
- Platelet count
- MCV
- MCH
- MCHC
- Other related indices

File:
cbc_augmented_with_ids.csv

---

### 2. HPLC Dataset
Contains hemoglobin variant analysis parameters:
- HbA
- HbA2
- HbF
- Other hemoglobin variants

File:
hplc_augmented_with_ids.csv

---

### 3. Scatter Dataset
Contains scatter plot derived parameters obtained from automated blood analyzers.

File:
scatter_augmented_with_ids.csv

---

## Methodology

The project follows a systematic machine learning workflow:

1. Data Collection
2. Data Cleaning and Preprocessing
3. Handling Missing Values
4. Feature Selection
5. Encoding of Labels
6. Train-Test Split
7. Model Training
8. Model Evaluation

---

## Algorithms Used

The following supervised learning algorithms were implemented:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Project Structure

blood_diagnosis_ml_project/
│
├── data/
│   ├── cbc_augmented_with_ids.csv
│   ├── hplc_augmented_with_ids.csv
│   └── scatter_augmented_with_ids.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   └── evaluation.py
│
└── README.md

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## How to Run the Project

1. Clone the repository:

   git clone https://github.com/jeshthabari/blood_diagnosis_ml_project.git

2. Navigate to the project directory:

   cd blood_diagnosis_ml_project

3. Install required dependencies:

   pip install -r requirements.txt

4. Run Jupyter Notebook:

   jupyter notebook

---

## Future Improvements

- Implementation of Deep Learning models
- Model deployment using Flask or Django
- Integration with hospital database systems
- Real-time prediction interface
- Explainable AI integration

---

## Author

Jeshtha Bari
,Divyesh Sahay  
BE Data Science  
Machine Learning Project
