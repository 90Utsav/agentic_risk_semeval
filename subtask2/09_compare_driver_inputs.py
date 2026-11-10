
"""
AgentRisk SemEval 2027
Subtask 2b - Experiment 2B-03

Compare three input representations:
1. Full interaction
2. Action only
3. Action emphasized

Model:
TF-IDF + One-vs-Rest Logistic Regression

Evaluation:
Three-fold scenario-grouped cross-validation
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
    precision_recall_fscore_support
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "subtask2/results"
FOLD_FILE = ROOT / "subtask1/configs/cv_folds.csv"

RESULTS.mkdir(parents=True, exist_ok=True)

NO_RISK = "No Material Risk"


# ----------------------------------------
# 1. Load data
# ----------------------------------------

print("Loading dataset...")

dataset = load_dataset(
    "AgentRisk/agentrisk-data-2027"
)

train = dataset["train"].to_pandas()

labels = [
    list(x) for x in train["risk_drivers"]
]

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)


# ----------------------------------------
# 2. Load existing CV assignments
# ----------------------------------------

fold_df = pd.read_csv(FOLD_FILE)

train = train.merge(
    fold_df[["example_id", "validation_fold"]],
    on="example_id",
    how="left",
    validate="one_to_one"
)

assert train["validation_fold"].notna().all()

folds = sorted(train["validation_fold"].unique())


# ----------------------------------------
# 3. Create three input representations
# ----------------------------------------

full = train["formatted_interaction"].fillna("").tolist()
actions = train["proposed_action"].fillna("").tolist()

emphasized = [
    context + (
        "\nACTION OF INTEREST:\n" + action
    ) * 2
    for context, action in zip(full, actions)
]

input_versions = {
    "full_interaction": full,
    "action_only": actions,
    "action_emphasized": emphasized,
}


# ----------------------------------------
# 4. Prediction decoding
# ----------------------------------------

def decode_probabilities(probabilities):
    classes = list(mlb.classes_)
    no_risk_idx = classes.index(NO_RISK)

    decoded = []

    for row in probabilities:

        selected = np.flatnonzero(
            row >= 0.5
        ).tolist()

        if not selected:
            selected = [int(np.argmax(row))]

        if no_risk_idx in selected and len(selected) > 1:

            others = [
                i for i in selected
                if i != no_risk_idx
            ]

            best_other = max(
                others,
                key=lambda i: row[i]
            )

            if row[no_risk_idx] >= row[best_other]:
                selected = [no_risk_idx]
            else:
                selected = others

        selected = sorted(
            selected,
            key=lambda i: row[i],
            reverse=True
        )[:2]

        decoded.append([
            classes[i] for i in selected
        ])

    return decoded


def jaccard(actual, predicted):
    a = set(actual)
    p = set(predicted)

    union = a | p

    return len(a & p) / len(union) if union else 1.0


# ----------------------------------------
# 5. Train and evaluate models
# ----------------------------------------

all_predictions = []
summary = []
per_label = []

for model_name, texts in input_versions.items():

    print(f"\n========== {model_name} ==========")

    model_predictions = []

    for fold in folds:

        train_idx = np.flatnonzero(
            train["validation_fold"].to_numpy() != fold
        )

        val_idx = np.flatnonzero(
            train["validation_fold"].to_numpy() == fold
        )

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True
        )

        X_train = vectorizer.fit_transform(
            [texts[i] for i in train_idx]
        )

        X_val = vectorizer.transform(
            [texts[i] for i in val_idx]
        )

        model = OneVsRestClassifier(
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            )
        )

        model.fit(X_train, y[train_idx])

        probabilities = model.predict_proba(X_val)

        predictions = decode_probabilities(
            probabilities
        )

        actuals = [
            labels[i] for i in val_idx
        ]

        fold_scores = [
            jaccard(a, p)
            for a, p in zip(actuals, predictions)
        ]

        print(
            f"Fold {fold} Jaccard:",
            round(float(np.mean(fold_scores)), 4)
        )

        for pos, idx in enumerate(val_idx):

            model_predictions.append({
                "model": model_name,
                "example_id": train.iloc[idx]["example_id"],
                "fold": int(fold),
                "actual": json.dumps(labels[idx]),
                "predicted": json.dumps(predictions[pos]),
                "jaccard": fold_scores[pos]
            })

    model_df = pd.DataFrame(model_predictions)

    assert len(model_df) == len(train)
    assert model_df["example_id"].is_unique

    all_predictions.extend(model_predictions)

    actual = [
        json.loads(x)
        for x in model_df["actual"]
    ]

    predicted = [
        json.loads(x)
        for x in model_df["predicted"]
    ]

    y_true = mlb.transform(actual)
    y_pred = mlb.transform(predicted)

    mean_jaccard = model_df["jaccard"].mean()

    exact_match = accuracy_score(y_true, y_pred)

    missed_risk = sum(
        NO_RISK in p and NO_RISK not in a
        for a, p in zip(actual, predicted)
    )

    false_alarms = sum(
        NO_RISK in a and NO_RISK not in p
        for a, p in zip(actual, predicted)
    )

    summary.append({
        "model": model_name,
        "oof_jaccard": mean_jaccard,
        "exact_match": exact_match,
        "missed_risks": missed_risk,
        "false_alarms": false_alarms
    })

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0
        )
    )

    for i, label in enumerate(mlb.classes_):
        per_label.append({
            "model": model_name,
            "label": label,
            "support": int(support[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i])
        })


# ----------------------------------------
# 6. Save results
# ----------------------------------------

summary_df = pd.DataFrame(summary)
predictions_df = pd.DataFrame(all_predictions)
per_label_df = pd.DataFrame(per_label)

summary_df.to_csv(
    RESULTS / "subtask2b_input_comparison.csv",
    index=False
)

predictions_df.to_csv(
    RESULTS / "subtask2b_input_comparison_predictions.csv",
    index=False
)

per_label_df.to_csv(
    RESULTS / "subtask2b_input_comparison_per_label.csv",
    index=False
)

print("\n========== FINAL COMPARISON ==========")

print(
    summary_df.round(4).to_string(index=False)
)

recall_table = per_label_df.pivot(
    index="label",
    columns="model",
    values="recall"
)

print("\n========== RECALL COMPARISON ==========")

print(
    recall_table.round(4).to_string()
)

print("\nExperiment 2B-03 completed!")
