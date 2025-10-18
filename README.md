# Loan-Approval-Prediction
Machine Learning task From DEPI for predicting the loan approval using Logestic Regrission and comparing between different penalties. Supervised by: ENG/ Baraa Abu Sallout 
---
# Project Report: Predicting Loan Approval Using Logistic Regression with Regularization and Hyperparameter Tuning

## 1. Introduction and Objective
This project aims to predict loan approval status for applicants based on a set of features, simulating a real-world banking scenario. We compare two machine learning algorithms—Logistic Regression (with L1 and L2 regularization) while applying hyperparameter tuning via GridSearchCV to optimize performance. The focus is on evaluating model effectiveness, the impact of regularization, and interpretability for practical use in loan approval processes.

The analysis was conducted in a Jupyter Notebook (Script.ipynb), which includes data loading, preprocessing, model training, evaluation, and visualizations.

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

### 3.2 Model Training (Without GridSearchCV)
- **Logistic Regression:**
  - L1 Penalty (Lasso regularization): Encourages sparsity (feature selection).
  - L2 Penalty (Ridge regularization): Penalizes large coefficients to reduce overfitting.
  - Default parameters otherwise (e.g., C=1.0).

### 3.3 Model Training (With GridSearchCV)
- **Hyperparameter Tuning:**
  - For Logistic Regression: Grid included `C` [0.01, 0.013, 0.02, 0.03, 0.05, 0.06, 0.07], `penalty` ['l1', 'l2', 'elasticnet'], `solver` ["liblinear", "newton-cg", "lbfgs", "sag", "saga"], `l1_ratio` [0.0, 0.5, 1.0] (for elasticnet).
- Used GridSearchCV with 5-fold cross-validation on the train set to find best parameters, then refit on full train set and evaluate on test set.
- Ensures balanced handling of class imbalance via stratified splits.

### 3.4 Comparison and Analysis
- Visualizations: Confusion matrices (via seaborn heatmap), bar plots of metrics (via plotly for interactive comparison).
- Effect of Regularization: L1 reduced coefficients for less important features (e.g., Self_Employed), while L2 smoothed the model without zeroing them out.
- Hyperparameter Tuning: Improved generalization, reducing overfitting (e.g., lower train-test gap).
- Number of Experiments Tried: 3 main experiments (Logistic L1, Logistic L2, Logistic GridSearch).

- Best Model Overall: Logistic Regression L1 (test accuracy: 0.854, F1: 0.901, AUC: 0.82). 
- When to Prefer Logistic: For interpretability (e.g., coefficient analysis) and linear patterns. Decision Trees for complex interactions but require pruning.

## 4. Results and Visualizations
- Performance Metrics Comparison (from Notebook's Plotly Bar Chart):
  - Logistic GridSearch led in all metrics (Accuracy: 0.82, Precision: 0.81, Recall: 0.95, F1: 0.88, AUC: 0.75).
- Confusion Matrices: Showed fewer false negatives in tuned models (critical for approvals).

## 5. Conclusion
The Logistic Regression model L1 without GridSearchCV performed the best overall, achieving a test accuracy of 0.854 and strong F1 score of 0.901, indicating reliable predictions with minimal overfitting.

**Repository Structure on GitHub:**
- `Script.ipynb`: Full implementation.
- `report.md`: This document.
- `Loan_Prediction.csv`: Dataset CSV (link to Kaggle provided).
