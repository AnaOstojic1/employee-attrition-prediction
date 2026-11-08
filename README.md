# Employee Attrition Prediction (Healthcare)

Predicting whether an employee will leave, using the IBM Watson healthcare attrition dataset
(1,620 employees after removing duplicates, about 12% left the company).

The notebook compares four models and focuses on the problems that come with imbalanced data.

## What's in the notebook

- data cleaning (removing the ID column, constant columns and duplicates)
- exploratory analysis (attrition rate by category, correlations, distributions)
- preprocessing inside a scikit-learn `Pipeline`, so the test data does not leak into training
- four models: logistic regression, random forest, MLP and a small Keras neural network
- class imbalance handled with `class_weight` (logistic regression, random forest, Keras) and oversampling (MLP)
- 5-fold stratified cross-validation and a decision threshold chosen from cross-validation, not from the test set
- bootstrap confidence intervals to check whether differences between models are real
- feature importance (permutation importance and logistic regression coefficients)

## Results

Cross-validation on the training set (baseline PR-AUC is 0.12, the share of employees who left):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic regression | 0.930 ± 0.014 | 0.726 ± 0.057 |
| Random forest | 0.907 ± 0.005 | 0.640 ± 0.070 |
| MLP | 0.915 ± 0.011 | 0.688 ± 0.065 |

PR-AUC on the test set (48 employees who left):

| Model | PR-AUC |
|---|---|
| Logistic regression | 0.774 |
| Keras neural network | 0.765 |
| MLP | 0.701 |
| Random forest | 0.676 |

Only the MLP is significantly worse than logistic regression (paired bootstrap, 95% interval).
The other differences are within noise, since the test set has few positive cases.
Logistic regression is the simplest model and is as good as the rest, so it is the final choice.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Healthcare_Churn_Prediction.ipynb
```

The notebook downloads the dataset from Kaggle (`jpmiller/employee-attrition-for-healthcare`) through `kagglehub`.
If you already have `watson_healthcare_modified.csv`, put it in the same folder and it will be used instead.

TensorFlow is only needed for the Keras part. Without it that section is skipped and the rest still runs.

## Notes

- The dataset is a demo dataset, so these results should not be applied directly to real HR decisions.
- Gender, age and marital status are used as features and can introduce bias. Set `EXCLUDE_SENSITIVE = True`
  in the notebook to leave out gender.
- The test set is small (48 employees who left), so small differences between models are not reliable.

## Tools

Python, pandas, scikit-learn, imbalanced-learn, TensorFlow/Keras, seaborn, matplotlib
