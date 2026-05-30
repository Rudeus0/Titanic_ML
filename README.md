# Titanic ML — Survival Prediction

Binary classification project predicting Titanic passenger survival using Logistic Regression, Random Forest, and XGBoost. Built as part of Phase 3 ML Core to learn classification metrics, feature encoding, and model comparison.

---

## Results

| Model | Accuracy | ROC AUC | F1 (Survived) | Recall (Survived) |
|-------|----------|---------|----------------|-------------------|
| Logistic Regression | 0.8101 | 0.8817 | 0.76 | 0.74 |
| **Random Forest** | **0.8268** | **0.8958** | **0.79** | **0.77** |
| XGBoost | 0.7989 | 0.8754 | 0.75 | 0.74 |

**Winner: Random Forest** — highest on all metrics.

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

## Pipeline Flow

```
load_data()           → read titanic_clean.csv
transform_and_scale() → one-hot encode Sex + Embarked, split 80/20, StandardScaler
train_model()         → Logistic Regression, Random Forest, XGBoost
evaluate_models()     → accuracy, confusion matrix, classification report, ROC AUC
```

---

## Key Findings

**Random Forest beat Logistic Regression** — 82.7% vs 81.0% accuracy. On a small dataset like Titanic (891 rows), the difference is small. This is expected — Titanic survival is largely driven by simple rules (women and children first, class matters) which a linear model can already capture well.

**Confusion Matrix Analysis — Best Model (Random Forest):**
```
[[91  14]   → 91 correctly predicted NOT survived, 14 false alarms
 [17  57]]  → 57 correctly predicted survived, 17 missed survivors
```

**Logistic Regression came very close** — survival patterns in Titanic are largely linear. Sex, Pclass, and Fare are strong predictors that a simple model captures well.

---