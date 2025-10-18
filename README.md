# Loan Prediction Analysis and Binary Classification Report Using Logistic Regression

## Introduction

This report summarizes the analysis of the Loan Prediction Dataset using a Jupyter Notebook titled **Logistic Regression for Binary Classification**. The primary goal is to build a binary classification model to predict loan approval (`Loan_Status`: Y or N) based on features such as income, education, and credit history.

Python libraries including Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn were used for processing and training. The dataset contains 614 samples and 13 features, with some missing values that were handled appropriately.

### Project Objectives:

- Understand the data structure and address issues.
- Engineer new features to improve performance.
- Train a Logistic Regression model with hyperparameter tuning.
- Evaluate the model and compare it with other models.

## Data Understanding

The data was loaded from the file `Loan_Predication.csv` (Note: Recommend correcting to `Loan_Prediction` for accuracy).

### Data Overview:

- **Number of Rows**: 614
- **Number of Columns**: 13
- **Data Types**:
  - Categorical: Gender, Married, Dependents, Education, Self_Employed, Property_Area, Loan_Status.
  - Numerical: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History.

| Column            | Type    | Non-Null Count | Brief Description                |
| ----------------- | ------- | -------------- | -------------------------------- |
| Loan_ID           | Object  | 614            | Loan ID (removed later)          |
| Gender            | Object  | 601            | Applicant's gender               |
| Married           | Object  | 611            | Marital status                   |
| Dependents        | Object  | 599            | Number of dependents             |
| Education         | Object  | 614            | Education level                  |
| Self_Employed     | Object  | 582            | Self-employed status             |
| ApplicantIncome   | Int64   | 614            | Applicant's income               |
| CoapplicantIncome | Float64 | 614            | Co-applicant's income            |
| LoanAmount        | Float64 | 592            | Loan amount                      |
| Loan_Amount_Term  | Float64 | 600            | Loan term (months)               |
| Credit_History    | Float64 | 564            | Credit history (1: Good, 0: Bad) |
| Property_Area     | Object  | 614            | Property area                    |
| Loan_Status       | Object  | 614            | Loan status (Y/N)                |

- **Missing Values**: Present in LoanAmount (22), Gender (13), Married (3), Dependents (15), Self_Employed (32), Loan_Amount_Term (14), Credit_History (50).
- **Duplicates**: No duplicate rows (0 duplicates).

### Target Variable Distribution (Loan_Status):

- Y (Approved): ~69% of samples.
- N (Rejected): ~31% of samples.
  This indicates a slight class imbalance, which may require handling in the model.

## Data Cleaning and Handling Missing Values

The following steps were taken:

1. **Remove Unnecessary Column**: `Loan_ID` (unique identifier, not useful for prediction).
2. **Impute Missing Values**:
   - LoanAmount: Using median = 128.
   - Categorical (Gender, Married, Dependents, Self_Employed): Using mode = 'Male', 'Yes', '0', 'No'.
   - Loan_Amount_Term: Using mode = 360.
   - Credit_History: Using mode = 1.0.
3. **Data Type Conversion**: `Dependents` from string ('3+') to integer (3).

After cleaning, all columns have no missing values (614 non-null each).

## Feature Engineering

A new feature was created to enhance the model:

- **Total_Income**: Sum of applicant's and co-applicant's income (`ApplicantIncome + CoapplicantIncome`).
  - Example: Row 1: 5849 + 0 = 5849.

This feature captures the overall financial capacity of the applicant.

## Data Encoding

Categorical variables were encoded using **One-Hot Encoding** to avoid ordinal assumptions:

- **Unique Values**:
  - Gender: ['Male', 'Female']
  - Married: ['No', 'Yes']
  - Education: ['Graduate', 'Not Graduate']
  - Self_Employed: ['No', 'Yes']
  - Property_Area: ['Urban', 'Rural', 'Semiurban']

`Loan_Status` was encoded to: Y=1, N=0.

## Data Splitting and Model Training

- **Data Split**: 80% training (train), 20% testing (test) using `train_test_split` (random_state=42).
- **Scaling**: `StandardScaler` applied to numerical features (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Total_Income).
- **Model**: Logistic Regression with **GridSearchCV** for hyperparameter tuning:
  - Searched Parameters: `C` (0.01 to 100), `penalty` ('l1', 'l2'), `solver` ('liblinear', 'lbfgs').
  - Best Parameters: C=1.0, penalty='l2', solver='liblinear'.
- **Training**: Model trained on training set and evaluated on test set.

## Model Evaluation

The model was evaluated using multiple metrics:

### 1. Accuracy:

- Training: 0.8214
- Testing: 0.8104

### 2. Classification Report:

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0 (N)            | 0.76      | 0.59   | 0.67     | 44      |
| 1 (Y)            | 0.83      | 0.90   | 0.86     | 121     |
| **Weighted Avg** | 0.81      | 0.81   | 0.81     | 165     |

### 3. Confusion Matrix:

- True Positives (TP): 109
- True Negatives (TN): 26
- False Positives (FP): 12
- False Negatives (FN): 18
- **Visualization**: A heatmap was plotted, showing strong performance in predicting approvals (Y).

### 4. ROC Curve and AUC:

- **AUC Score**: 0.8660 (High value indicates good discrimination between classes).
- **Visualization**: ROC curve plotted against the random classifier (0.5).

## Model Comparison

Logistic Regression was compared with other models using test accuracy:

| Model               | Test Accuracy |
| ------------------- | ------------- |
| Logistic Regression | 0.8104        |
| Random Forest       | 0.7879        |
| SVM                 | 0.7727        |
| KNN                 | 0.7424        |
| Decision Tree       | 0.7152        |

- **Visualization**: Bar chart showing Logistic Regression's superiority in accuracy.
- **Conclusion**: Logistic Regression performs best for this dataset due to its simplicity and effectiveness with mixed categorical/numerical data.

## Conclusion and Recommendations

- **Overall Performance**: The model is successful (accuracy ~81%), but it shows slight bias toward the majority class (Y), suggesting techniques like SMOTE for balancing.
- **Limitations**: Small dataset (614 samples); additional features (e.g., income-to-loan ratio) could be beneficial.
- **Recommendations**:
  - Implement Cross-Validation for stability.
  - Experiment with advanced models like XGBoost.
  - Deploy the model as an API for real-world use.

For more details, refer to the original Notebook. If you have additional questions, let me know!

**Report Date**: October 18, 2025  
**Prepared by**: Grok (based on the provided Notebook)
