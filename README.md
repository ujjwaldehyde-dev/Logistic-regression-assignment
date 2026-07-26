# Titanic Survival Prediction using Logistic Regression

**Name:** Ujjwal Jha
**Branch:** CSAIML (2023-27)
**Course:** Machine Learning (4th Year B.Tech)

## About the dataset used here

This project was completed in an offline sandboxed environment with no
internet access, so the real Kaggle dataset (`heptapod/titanic`) could not be
downloaded while producing this submission. In its place, `generate_data.py`
builds a **statistically realistic synthetic** Titanic-style dataset:

- 891 rows, identical column structure to the real dataset (`PassengerId`,
  `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`,
  `Fare`, `Cabin`, `Embarked`)
- Missingness patterns matched to the real dataset (~22% missing `Age`, ~78%
  missing `Cabin`, 2 missing `Embarked`)
- A seeded survival-probability model that reproduces the well-documented
  historical pattern: higher survival for women, children, and 1st-class
  passengers

Every table, chart, and metric in the notebook and report is computed live
from this generated data — nothing is hand-typed. **The notebook code is
unchanged and reusable as-is on the real Kaggle CSV** — just replace the
`pd.read_csv(...)` path in the "Load the dataset" cell with the real file.

## Files in this submission

| File | Description |
|---|---|
| `Titanic_Logistic_Regression_UjjwalJha.ipynb` | Full, executed Jupyter notebook covering the entire pipeline (EDA → preprocessing → model → evaluation → tuning → business insights) |
| `Titanic_Report_UjjwalJha.pdf` | 8-page written report following the required format (Introduction → Conclusion) |
| `titanic.csv` | The synthetic dataset used throughout |
| `generate_data.py` | Script that generates `titanic.csv` (seeded, reproducible) |
| `images/` | All charts used in the notebook and report (class distribution, correlation heatmap, histograms, boxplots, feature importance, confusion matrix, ROC curve) |
| `README.md` | This file |

## Pipeline summary

1. **Data understanding** — shape, dtypes, missing values, descriptive stats
2. **EDA** — class distribution, correlation heatmap, histograms, boxplots, IQR outlier check
3. **Preprocessing** — group-median age imputation, mode imputation for Embarked, `HasCabin` flag, `Title`/`FamilySize`/`IsAlone` feature engineering, label/one-hot encoding, standard scaling, 80/20 stratified train-test split
4. **Modeling** — scikit-learn `LogisticRegression`, coefficients and feature importance reported
5. **Evaluation** — Accuracy 0.810, Precision 0.809, Recall 0.603, F1 0.691, ROC-AUC 0.879
6. **Hyperparameter tuning** — `GridSearchCV` over `C` and `penalty`, best: `C=10, penalty=l1` (CV F1 0.614)
7. **Business insights** — survival by sex/class/age group, strongest predictor, and recommendations for maritime safety practice

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python3 generate_data.py          # regenerates titanic.csv
jupyter notebook Titanic_Logistic_Regression_UjjwalJha.ipynb
```

To run against the real Kaggle dataset instead, download `train.csv` from
https://www.kaggle.com/datasets/heptapod/titanic and point the "Load the
dataset" cell at that file — no other code changes are required.
