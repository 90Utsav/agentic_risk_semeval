
"""
AgentRisk SemEval 2027
Subtask 2b - Experiment 2B-02

TF-IDF + One-vs-Rest Logistic Regression

Evaluation:
- Same saved scenario-grouped folds as Subtask 1/2a
- Mean example-wise Jaccard
- Exact-match accuracy
- Per-label precision, recall and F1
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# ========================================
# 1. PATHS
# ========================================

ROOT = Path(__file__).resolve().parents[1]

FOLD_FILE = ROOT / "subtask1/configs/cv_folds.csv"
RESULTS = ROOT / "subtask2/results"

RESULTS.mkdir(parents=True, exist_ok=True)

NO_RISK = "No Material Risk"


# ========================================
# 2. LOAD DATA
# ========================================

print("Loading AgentRisk dataset...")

dataset = load_dataset(
    "AgentRisk/agentrisk-data-2027"
)

train = dataset["train"].to_pandas()

labels = [
    list(x)
    for x in train["risk_drivers"]
]

texts = train["formatted_interaction"].fillna("").tolist()

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)

print("Training examples:", len(train))
print("Risk-driver classes:", list(mlb.classes_))


# ========================================
# 3. LOAD EXISTING CV FOLDS
# ========================================

if not FOLD_FILE.exists():
    raise FileNotFoundError(
        f"Saved CV folds not found: {FOLD_FILE}"
    )

fold_df = pd.read_csv(FOLD_FILE)

print("\nSaved fold columns:", fold_df.columns.tolist())

possible_columns = [
    "fold",
    "cv_fold",
    "fold_id",
    "validation_fold",
    "val_fold",
]

fold_column = next(
    (c for c in possible_columns if c in fold_df.columns),
    None
)

if fold_column is None:
    raise ValueError(
        "Could not identify fold column. "
        "Inspect subtask1/configs/cv_folds.csv."
    )

if "example_id" not in fold_df.columns:
    raise ValueError("Fold file must contain example_id.")

if fold_df["example_id"].duplicated().any():
    raise ValueError("Duplicate example IDs in fold file.")

fold_lookup = fold_df.set_index("example_id")[fold_column]

train["fold"] = train["example_id"].map(fold_lookup)

if train["fold"].isna().any():
    raise ValueError(
        "Some training examples are missing saved fold assignments."
    )

folds = sorted(train["fold"].unique())

if len(folds) != 3:
    raise ValueError(
        f"Expected 3 folds, but found {len(folds)}: {folds}"
    )

print("Using saved folds:", folds)


# ========================================
# 4. DECODE PREDICTIONS
# ========================================

def decode_probabilities(probs, classes):
    """
    Predict 1-2 labels with a fixed threshold.

    Rules:
    - Threshold = 0.5
    - If none selected, use highest probability
    - No Material Risk cannot co-occur
    - Maximum two labels
    """

    predictions = []
    classes = list(classes)

    no_risk_idx = classes.index(NO_RISK)

    for row in probs:

        selected = np.flatnonzero(row >= 0.5).tolist()

        if not selected:
            selected = [int(np.argmax(row))]

        if no_risk_idx in selected and len(selected) > 1:
            other_indices = [
                i for i in selected
                if i != no_risk_idx
            ]

            best_other = max(
                other_indices,
                key=lambda i: row[i]
            )

            if row[no_risk_idx] >= row[best_other]:
                selected = [no_risk_idx]
            else:
                selected = other_indices

        selected = sorted(
            selected,
            key=lambda i: row[i],
            reverse=True
        )[:2]

        predictions.append([
            classes[i] for i in selected
        ])

    return predictions


def jaccard(actual, predicted):
    a = set(actual)
    p = set(predicted)

    union = a | p

    return len(a & p) / len(union) if union else 1.0


# ========================================
# 5. GROUPED CROSS-VALIDATION
# ========================================

all_predictions = []

for fold in folds:

    print(f"\n========== FOLD {fold} ==========")

    train_idx = np.flatnonzero(
        train["fold"].to_numpy() != fold
    )

    val_idx = np.flatnonzero(
        train["fold"].to_numpy() == fold
    )

    X_train = [texts[i] for i in train_idx]
    X_val = [texts[i] for i in val_idx]

    y_train = y[train_idx]

    print("Train size:", len(train_idx))
    print("Validation size:", len(val_idx))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = OneVsRestClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    )

    model.fit(X_train_tfidf, y_train)

    probabilities = model.predict_proba(
        X_val_tfidf
    )

    predicted = decode_probabilities(
        probabilities,
        mlb.classes_
    )

    actual = [labels[i] for i in val_idx]

    scores = [
        jaccard(a, p)
        for a, p in zip(actual, predicted)
    ]

    print(
        "Fold Jaccard:",
        round(float(np.mean(scores)), 4)
    )

    for position, idx in enumerate(val_idx):

        all_predictions.append({
            "example_id": train.iloc[idx]["example_id"],
            "fold": fold,
            "actual": json.dumps(list(labels[idx])),
            "predicted": json.dumps(predicted[position]),
            "jaccard": scores[position]
        })


# ========================================
# 6. OVERALL EVALUATION
# ========================================

results = pd.DataFrame(all_predictions)

if len(results) != len(train):
    raise ValueError("Incorrect number of OOF predictions.")

if results["example_id"].duplicated().any():
    raise ValueError("Duplicate OOF predictions found.")

actual = results["actual"].apply(json.loads).tolist()
predicted = results["predicted"].apply(json.loads).tolist()

y_true = mlb.transform(actual)
y_pred = mlb.transform(predicted)

mean_jaccard = results["jaccard"].mean()
exact_match = accuracy_score(y_true, y_pred)

missed_risk = sum(
    NO_RISK in p and NO_RISK not in a
    for a, p in zip(actual, predicted)
)

false_alarms = sum(
    NO_RISK in a and NO_RISK not in p
    for a, p in zip(actual, predicted)
)

print("\n========== FINAL OOF RESULTS ==========")

print("Mean Jaccard:", round(mean_jaccard, 4))
print("Exact-match accuracy:", round(exact_match, 4))
print("Missed risks:", missed_risk)
print("False alarms:", false_alarms)


# ========================================
# 7. PER-LABEL METRICS
# ========================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )
)

metrics = pd.DataFrame({
    "label": mlb.classes_,
    "support": support,
    "precision": precision,
    "recall": recall,
    "f1": f1
})

print("\n========== PER-LABEL PERFORMANCE ==========")

print(
    metrics.round(4).to_string(index=False)
)


# ========================================
# 8. SAVE RESULTS
# ========================================

results.to_csv(
    RESULTS / "subtask2b_baseline_cv.csv",
    index=False
)

metrics.to_csv(
    RESULTS / "subtask2b_baseline_per_label.csv",
    index=False
)

print("\nResults saved successfully!")
