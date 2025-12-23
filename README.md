# Loan-Approval-Prediction
Machine Learning task From DEPI for predicting the loan approval using Logestic Regrission and comparing between different penalties. Supervised by: ENG/ Baraa Abu Sallout 
---
# Project Report: Predicting Loan Approval Using Logistic Regression with Regularization and Hyperparameter Tuning

## 1. Introduction and Objective
This project builds a binary classification model to predict whether a loan application will be **approved (Y)** or **rejected (N)** based on applicant features.  

We implement and compare:
- Logistic Regression with L1 (Lasso) and L2 (Ridge) regularization
- Hyperparameter-tuned Logistic Regression (GridSearchCV)
- Decision Tree Classifier (baseline & tuned)
- Interactive performance comparison using Plotly

The goal is to achieve high accuracy and interpretability while understanding the impact of regularization and tree pruning in a real-world banking use case.

The analysis was conducted in a Jupyter Notebook (Loan Approval Prediction.ipynb), which includes data loading, preprocessing, model training, evaluation, and visualizations.

## 2. Dataset Description
The dataset used is the "Loan Prediction Problem Dataset" from Kaggle (available at [https://www.kaggle.com/datasets/ninzaami/loan-predication/data](https://www.kaggle.com/datasets/ninzaami/loan-predication/data)). It consists of 614 records (loan applications) with 13 columns, representing a mix of categorical and numerical features. The target variable is `Loan_Status` (Y for approved, N for not approved), making this a binary classification problem.

### Key Columns and Their Meanings:
| Column Name       | Meaning                  | Type         | Explanation |
|-------------------|--------------------------|--------------|-------------|
| Loan_ID           | Loan Identifier          | Object      | Unique ID (dropped during training). |
| Gender            | Applicant’s Gender       | Categorical | Male/Female (missing values: 13). |
| Married           | Marital Status           | Categorical | Yes/No (missing values: 3). |
| Dependents        | Number of Dependents     | Categorical | 0/1/2/3+ (treated as categorical due to "3+"; missing values: 15). |
| Education         | Education Level          | Categorical | Graduate/Not Graduate. |
| Self_Employed     | Employment Type          | Categorical | Yes/No (missing values: 32). |
| ApplicantIncome   | Applicant’s Income       | Numerical   | Monthly income (mean: ~5403, range: 150–81000). |
| CoapplicantIncome | Co-applicant’s Income    | Numerical   | Monthly co-applicant income (mean: ~1621, often 0). |
| LoanAmount        | Loan Amount              | Numerical   | Amount requested in thousands (mean: ~146, missing values: 22). |
| Loan_Amount_Term  | Loan Term                | Numerical   | Term in months (mean: ~342, often 360; missing values: 14). |
| Credit_History    | Credit History           | Numerical   | 1 (good)/0 (bad) (missing values: 50; strong predictor). |
| Property_Area     | Property Location        | Categorical | Urban/Semiurban/Rural. |
| Loan_Status       | Target Variable          | Categorical | Y (approved)/N (not approved; ~69% approved). |

**Notes on Dataset:**
- Imbalanced target (more approvals than rejections).
- Missing values in several columns require imputation.
- Real-world insights: Approvals correlate with high income, good credit history, and urban/semiurban properties.
- A new feature, `TotalIncome` (ApplicantIncome + CoapplicantIncome), was engineered to capture combined financial strength.

## 3. Methodology
### 3.1 Data Preprocessing
- **Loading and Investigation:** Data loaded from `Loan_Prediction.csvv` using pandas. Used `df.info()` to identify data types and missing values, and `df.describe()` for summary statistics (e.g., ApplicantIncome mean: 5403, std: 6109; outliers in income noted).
- **Handling Missing Values:**
  - Categorical (Gender, Married, Dependents, Self_Employed): Imputed with mode (most frequent value).
  - Numerical (LoanAmount, Loan_Amount_Term): Imputed with median to handle skewness.
  - Credit_History: Imputed with mode (1, as it's a strong predictor).
- **Encoding Categorical Features:** Used OneHotEncoder for multi-category features (e.g., Property_Area) and LabelEncoder for binary (e.g., Gender, Married, Education, Self_Employed). Dependents treated as categorical.
- **Feature Engineering:** Created `TotalIncome`. Dropped `Loan_ID`. Applied log transformation to skewed features like LoanAmount.
- **Feature Scaling:** StandardScaler applied to numerical features (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term) for Logistic Regression sensitivity.
- **Data Split:** 70/30 train-test split using `train_test_split` from scikit-learn (random_state=42 for reproducibility).

Libraries used: pandas, numpy, seaborn, matplotlib, scikit-learn (for preprocessing, models, metrics, and GridSearchCV), plotly (for visualizations).

#### 3.2 Models Implemented
| Model                            | Regularization / Tuning                     | Key Hyperparameters Searched                              |
|-----------------------------------|----------------------------------------------|--------------------------------------------------------------------|
| Logistic Regression L1            | Lasso (penalty='l1')                         | Default C=1.0                                                      |
| Logistic Regression L2            | Ridge (penalty='l2')                         | Default C=1.0                                                      |
| Logistic Regression + GridSearch  | L1 / L2 / ElasticNet                         | C, penalty, solver, l1_ratio                                       |
| Decision Tree (baseline)          | –                                            | Default settings                                                   |
| Decision Tree + GridSearch        | Pruning via depth & sample constraints       | max_depth, min_samples_split, min_samples_leaf, criterion, splitter |

GridSearchCV used 5-fold CV on training set.
### 3.3 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- AUC-ROC
- Confusion Matrix
## 4. Results Summary

| Model                            | Accuracy | Precision | Recall | F1-Score | AUC   |
|----------------------------------|----------|-----------|--------|----------|-------|
| Logistic Regression L1 (default) | **0.862** | 0.840     | 0.988  | **0.908** | 0.862 |
| Logistic Regression L2 (default) | 0.797    | 0.794     | 0.953  | 0.866    | 0.833 |
| Logistic Regression (GridSearch) | **0.862** | 0.840     | 0.988  | **0.908** | 0.862 |
| Decision Tree (default)          | 0.772    | 0.821     | 0.871  | 0.845    | 0.735 |
| Decision Tree (GridSearch)       | 0.854    | 0.831     | 0.976  | 0.897    | 0.812 |

**Best Model:** **Logistic Regression with L1 regularization (C ≈ 0.06 after tuning)**  
→ Highest accuracy & F1, excellent recall (catches almost all approvable loans with very few false negatives).

## 5. Key Insights
- `Credit_History` is by far the most important feature.
- L1 regularization successfully performs feature selection (e.g., zeros out `Self_Employed`).
- Decision Trees overfit heavily without tuning but reach strong performance after pruning.
- TotalIncome proved more predictive than separate income columns.
- All models benefit significantly from hyperparameter tuning.

## 6. Repository Structure
```
Loan-Approval-Prediction/
├── Loan Approval Prediction.ipynb    # Complete notebook with code & visualizations
├── Loan_Prediction.csv               # Dataset
├── README.md                         # This file
└── requirements.txt (optional)
```
## 7. How to Run
```bash
pip install -r requirements.txt
---

## Conclusion
The tuned Logistic Regression (L1) model achieves 86.2% accuracy and F1 = 0.908, making it the most reliable and interpretable choice for deployment in a loan approval system.
Feel free to fork, improve, or deploy!
Happy coding!
