# Student Performance & Placement Predictor

An end-to-end Machine Learning and Streamlit dashboard project for analyzing student academic performance and predicting placement outcomes from student-related academic and skill attributes.

## 📌 Project Overview

The **Student Performance & Placement Predictor** combines data analysis, machine learning, model evaluation, and an interactive Streamlit dashboard into a single academic analytics project.

The project focuses on two separate ML tasks:

1. **Student Performance Prediction** — predicts a student's final academic score.
2. **Placement Prediction** — predicts whether a student is likely to be placed based on academic and skill-related attributes.

The project is designed as a practical demonstration of the complete machine-learning workflow:

**Data → Cleaning → Exploration → Feature Preparation → Model Training → Evaluation → Prediction → Dashboard**

---

## 🎯 Objectives

* Analyze student academic and behavioral data.
* Identify patterns associated with academic performance.
* Predict final academic performance using regression.
* Analyze factors associated with placement outcomes.
* Predict placement outcomes using classification.
* Compare baseline and tree-based ML models.
* Provide an interactive dashboard for exploration and prediction.
* Demonstrate an end-to-end Data Science workflow.

---

## 🚀 Key Features

### 📊 Dataset & EDA

* Dataset overview
* Descriptive statistics
* Missing-value analysis
* Duplicate-value analysis
* Categorical feature exploration
* Correlation analysis
* Academic performance analysis
* Placement distribution analysis

### 🎓 Performance Analysis

* Academic performance exploration
* Study-time analysis
* Failure-history analysis
* Absence-related analysis
* Feature relationships with final academic score

### 🤖 Performance Prediction

Users can enter student information and receive a predicted final academic score.

The model uses the saved preprocessing pipeline and trained regression model rather than generating a rule-based prediction.

### 💼 Placement Prediction

Users can enter student academic and skill-related information including:

* IQ
* Previous semester result
* CGPA
* Academic performance
* Internship experience
* Extra-curricular score
* Communication skills
* Projects completed

The system returns a placement prediction and model probability.

### 📈 Model Evaluation

The dashboard presents:

* MAE
* RMSE
* R²
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion matrix

### 💡 Insights

The dashboard provides data-driven observations from the analyzed datasets.

---

# 🧠 Machine Learning Approach

## 1. Student Performance Prediction

### Problem Type

**Regression**

### Target

`G3` — final academic score from the selected UCI Student Performance dataset.

### Models

* Linear Regression — baseline
* Random Forest Regressor — final saved model

### Evaluation

| Model                   |    MAE |   RMSE |     R² |
| ----------------------- | -----: | -----: | -----: |
| Linear Regression       | 2.1564 | 2.8618 | 0.1602 |
| Random Forest Regressor | 2.0452 | 2.8166 | 0.1865 |

The Random Forest Regressor was saved as the performance prediction model.

### Important Modeling Consideration

The final academic score is treated as the prediction target. The target itself is not used as an input feature.

Earlier-grade variables such as G1 and G2 require careful consideration because they may introduce target leakage depending on the intended prediction point. The current dashboard therefore uses the available student feature dataset without treating the final target as an input.

---

# 💼 2. Placement Prediction

### Problem Type

**Binary Classification**

### Target

The placement dataset contains:

* `No` → 0
* `Yes` → 1

### Features Used

The clean placement model uses:

* IQ
* Previous Semester Result
* CGPA
* Academic Performance
* Internship Experience
* Extra-Curricular Score
* Communication Skills
* Projects Completed

`College_ID` was excluded from the clean model because it identifies the college rather than representing a meaningful student-level predictive characteristic.

### Models

* Logistic Regression — baseline
* Random Forest Classifier — saved model

### Logistic Regression Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9035 |
| Precision | 0.7769 |
| Recall    | 0.5873 |
| F1 Score  | 0.6690 |
| ROC-AUC   | 0.9451 |

### Random Forest Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9995 |
| Precision | 1.0000 |
| Recall    | 0.9970 |
| F1 Score  | 0.9985 |
| ROC-AUC   | 1.0000 |

Confusion matrix:

```text
[[1668    0]
 [   1  331]]
```

### ⚠️ Important Interpretation

The unusually high Random Forest performance is **dataset-specific** and should not be interpreted as guaranteed real-world placement accuracy.

A model with near-perfect performance on a particular dataset requires additional investigation into the dataset's construction, feature-target relationships, and possible hidden patterns before making real-world claims.

The project therefore presents these results as experimental model evaluation rather than as a guarantee of actual placement outcomes.

---

# 📚 Datasets

## Student Performance Dataset

The academic-performance pipeline uses the **UCI Student Performance dataset**.

Dataset characteristics used in this project:

* 649 student records
* 30 feature columns in the feature dataset
* Academic target information including G1, G2 and G3
* Numerical and categorical variables
* No missing values detected in the loaded feature dataset
* No duplicate rows detected

The final academic score `G3` is used as the regression target.

## Placement Dataset

The placement pipeline uses the **College Student Placement Factors Dataset**.

Dataset characteristics used in this project:

* 10,000 student records
* Placement target: Yes / No
* Academic and skill-related student attributes
* 8 clean predictive features used by the final model
* Placement distribution:

```text
No  : 8341
Yes : 1659
```

The two datasets are treated as **separate datasets and separate ML pipelines**. They are not represented as one unified student population.

---

# 🔄 Data Processing Workflow

```text
Raw Dataset
     ↓
Data Validation
     ↓
Missing & Duplicate Checks
     ↓
Exploratory Data Analysis
     ↓
Feature Selection
     ↓
Categorical Encoding
     ↓
Numerical Scaling
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
Streamlit Dashboard
     ↓
Interactive Prediction
```

---

# 🏗️ Project Structure

```text
Student-Performance-Placement-Predictor/
│
├── data/
│   ├── raw/
│   │   ├── student_performance.csv
│   │   ├── student_performance_targets.csv
│   │   └── placement_data.csv
│   │
│   └── processed/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│
├── models/
│   ├── performance_model.joblib
│   ├── performance_preprocessor.joblib
│   ├── placement_model.joblib
│   └── placement_preprocessor.joblib
│
├── dashboard/
│   └── app.py
│
├── reports/
│   └── figures/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🖥️ Dashboard Pages

The Streamlit application currently contains:

1. **Overview**
2. **Dataset & EDA**
3. **Performance Analysis**
4. **Performance Prediction**
5. **Placement Prediction**
6. **Model Evaluation**
7. **Insights**
8. **About**

---

# ⚙️ Installation

## 1. Clone the repository

Clone this repository to your local machine.

## 2. Open the project directory

```powershell
cd Student-Performance-Placement-Predictor
```

## 3. Create a virtual environment

```powershell
python -m venv .venv
```

## 4. Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

## 5. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

# ▶️ Run the Dashboard

From the project root:

```powershell
streamlit run dashboard/app.py
```

The application will open in your browser through the local Streamlit server.

---

# 🧪 Testing

Pytest is included in the development environment.

Run:

```powershell
python -m pytest
```

The current project does not yet contain automated test cases, so pytest currently reports that no tests are collected.

---

# 💾 Saved ML Artifacts

The trained models and preprocessing pipelines are stored using Joblib.

### Performance

```text
models/performance_model.joblib
models/performance_preprocessor.joblib
```

### Placement

```text
models/placement_model.joblib
models/placement_preprocessor.joblib
```

The dashboard loads these saved artifacts to perform predictions.

---

# 📊 Evaluation Metrics

### Regression

The performance model is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### Classification

The placement models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

---

# ⚠️ Limitations

This project is an educational and analytical ML application and has several limitations.

### 1. Dataset limitations

The academic and placement datasets represent different populations and should not be interpreted as one combined institutional dataset.

### 2. Generalization

Model performance on public datasets does not guarantee equivalent performance on a real college or institutional dataset.

### 3. Placement prediction

Placement is influenced by many factors that may not be represented in the selected dataset.

### 4. Probability interpretation

The placement probability shown by the model is a model-derived estimate, not a guarantee of actual placement.

### 5. Correlation vs causation

Observed relationships in the data should not automatically be interpreted as causal relationships.

### 6. High placement-model performance

The near-perfect Random Forest placement result should be investigated further before any real-world deployment or decision-making use.

---

# 🔮 Future Improvements

Potential future improvements include:

* Add automated unit and integration tests.
* Create a dedicated preprocessing pipeline module in `src/`.
* Add more robust model validation.
* Investigate potential data leakage and dataset-generation patterns.
* Perform hyperparameter tuning where justified.
* Add cross-validation to the complete model comparison workflow.
* Add model explainability using permutation importance or SHAP.
* Add downloadable prediction reports.
* Add model monitoring for future institutional data.
* Add authentication for institutional deployment.
* Evaluate models using genuine anonymized institutional data.
* Improve dashboard navigation and interactive visualizations.
* Add automated data validation before prediction.

---

# 🛡️ Responsible Use

This project should be used as a **decision-support and educational analytics system**, not as an automated decision-maker.

Predictions should not be used as the sole basis for:

* academic decisions,
* student exclusion,
* recruitment decisions,
* placement guarantees,
* or other high-impact decisions.

Real-world deployment would require appropriate institutional validation, privacy protection, fairness assessment, monitoring, and human oversight.

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Serialization

* Joblib

### Dashboard

* Streamlit

### Development

* VS Code
* Jupyter Notebook
* Pytest

---

# 👨‍💻 Project Status

**Current status: Functional ML + Interactive Dashboard**

The core workflow has been implemented and tested locally:

```text
Dataset
   ↓
EDA
   ↓
Preprocessing
   ↓
ML Models
   ↓
Evaluation
   ↓
Saved Models
   ↓
Streamlit Dashboard
   ↓
Interactive Predictions
```

---

## 📌 Learning Outcomes

This project demonstrates practical experience with:

* Python for Data Science
* Data cleaning and validation
* Exploratory Data Analysis
* Feature engineering and preprocessing
* Regression
* Classification
* Model evaluation
* Model serialization
* Streamlit application development
* Interactive ML prediction
* GitHub project organization
* Responsible interpretation of ML results

---

## ⭐ Project Goal

The goal of this project is to demonstrate how machine learning can be applied to student-related data to support **academic analysis and placement-oriented insights** while maintaining appropriate awareness of dataset limitations, prediction uncertainty, and responsible use.
