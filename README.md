# Logistic Regression for Binary Classification Report

## Executive Summary

This report presents a logistic regression analysis for loan prediction, comparing three model configurations (L1, L2, and optimized hyperparameters) on a dataset of 614 loan applications.

---

## 1. Dataset Overview

### Basic Information

- **Total Records**: 614 entries
- **Features**: 12 predictive features + 1 target variable
- **Target**: Loan_Status (Approved/Rejected)
- **Split**: 70% training (429), 30% testing (185)

### Feature Types

**Categorical**: Gender, Married, Dependents, Education, Self_Employed, Property_Area

**Numerical**: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History

---

## 2. Data Preprocessing

### Missing Values Treatment

| Feature               | Missing | Solution          |
| --------------------- | ------- | ----------------- |
| LoanAmount            | 22      | Median imputation |
| Credit_History        | 50      | Mode imputation   |
| Gender, Married, etc. | Various | Mode imputation   |

### Key Preprocessing Steps

1. No duplicates found
2. Removed Loan_ID column
3. Converted Dependents '3+' → 3
4. Created `Total_Income` feature
5. One-hot encoded categorical variables
6. Applied StandardScaler normalization

---

## 3. Model Development

Three logistic regression models were trained and evaluated:

1. **L2 Regularization** (Ridge)
2. **L1 Regularization** (Lasso)
3. **GridSearchCV Optimized** (Best hyperparameters)

### GridSearch Configuration

- **Parameters Tested**: C=[0.1, 1, 10], penalty=[l1, l2]
- **Best Configuration**: C=0.1, penalty=l1
- **Cross-Validation**: 5-fold stratified CV

---

## 4. Results & Model Comparison

### Performance Metrics

| Model   | Train Acc | Test Acc   | AUC        | Status |
| ------- | --------- | ---------- | ---------- | ------ |
| LR L1   | 79.49%    | **84.86%** | 0.8234     | ✓      |
| LR L2   | 79.49%    | **84.86%** | 0.8228     | ✓      |
| Best LR | 79.25%    | **84.86%** | **0.8262** | ✓      |

### Confusion Matrix (All Models - Identical)

```
                Predicted
              No      Yes
Actual
No            33       25
Yes            3      124
```

### Classification Metrics

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Rejected (0)** | 0.92      | 0.57   | 0.70     | 58      |
| **Approved (1)** | 0.83      | 0.98   | 0.90     | 127     |
| **Accuracy**     | -         | -      | **0.85** | 185     |

---

## 5. Key Findings

### Strengths

- **High Test Accuracy**: 84.86% across all models
- **Excellent Recall for Approvals**: 98% (identifies most valid loans)
- **No Overfitting**: Training accuracy (≈79%) < Test accuracy
- **Good AUC Score**: 0.82-0.83 indicates strong discrimination

### Limitations

- **25 False Positives**: Approved loans that should be rejected
- **3 False Negatives**: Rejected loans that should be approved

### Business Impact

**Type I Error (False Positive)**: 25 cases

- Risk: Potential loan defaults
- Impact: Financial loss

**Type II Error (False Negative)**: 3 cases

- Risk: Lost business opportunities
- Impact: Revenue loss

---

## 6. Conclusions

### Main Insights

1. **Consistent Performance**: All three models achieve identical test accuracy
2. **Regularization Effect**: L1 and L2 perform similarly; stronger regularization (C=0.1) is beneficial
3. **Conservative Model**: High approval recall suggests conservative risk approach
4. **Production Ready**: Best LR model shows stable, reliable predictions
