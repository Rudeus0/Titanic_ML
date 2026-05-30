# Titanic ML — Survival Prediction

Binary classification project predicting Titanic passenger survival using Logistic Regression, Random Forest, and XGBoost. Built as part of Phase 3 ML Core to learn classification metrics, feature encoding, and model comparison.

---

## Project Structure

```
Titanic_ML/
├── data/
│   └── titanic_clean.csv        ← cleaned data from P2 EDA project
├── notebooks/
│   └── analysis.ipynb           ← EDA, feature engineering, model exploration
├── plots/
├── src/
│   ├── train.py                 ← load, transform, scale, train
│   └── evaluate.py              ← metrics evaluation for all 3 models
├── main.py                      ← pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source:** Titanic passenger data (reused cleaned CSV from P2 EDA project)
- **Size:** 891 rows × 12 columns (raw) → 9 features after cleaning
- **Target:** `Survived` — 0 = did not survive, 1 = survived
- **Class distribution:** 549 (62%) did not survive, 342 (38%) survived

---

## Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `Pclass` | Numeric | Passenger class (1/2/3) |
| `Age` | Numeric | Passenger age |
| `SibSp` | Numeric | Siblings/spouses aboard |
| `Parch` | Numeric | Parents/children aboard |
| `Fare` | Numeric | Ticket fare |
| `FamilySize` | Numeric | SibSp + Parch + 1 |
| `Sex` | Encoded | male=1, female=0 |
| `Embarked` | Encoded | Port of embarkation |

**Dropped:** `PassengerId`, `Name`, `Ticket` — no predictive value.

---
