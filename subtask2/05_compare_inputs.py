\
"""
AgentRisk SemEval 2027
Subtask 2a — Experiment 2A-04

Compare three text input representations:
1. Full interaction
2. Proposed action only
3. Full interaction with proposed action emphasized

Model: TF-IDF + One-vs-Rest Logistic Regression
Evaluation: 3-fold grouped cross-validation
Primary metric: Mean Jaccard Similarity
"""

import json
import numpy as np
import pandas as pd

from pathlib import Path
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading AgentRisk dataset...")

dataset = load_dataset(
    "AgentRisk/agentrisk-data-2027"
)

train = dataset["train"].to_pandas()

print("Training examples:", len(train))


# ==========================================
# 2. LOAD EXISTING CROSS-VALIDATION FOLDS
# ==========================================

fold_path = Path(
    "subtask1/configs/cv_folds.csv"
)

if not fold_path.exists():
    raise FileNotFoundError(
        f"Cross-validation file not found: {fold_path}"
    )

saved_folds = pd.read_csv(fold_path)

assert saved_folds["example_id"].is_unique, (
    "Duplicate example IDs in saved folds!"
)

fold_map = saved_folds.set_index(
    "example_id"
)["validation_fold"]

fold_ids = train["example_id"].map(fold_map)

assert fold_ids.notna().all(), (
    "Some training examples have no fold assignment!"
)

fold_ids = fold_ids.astype(int).to_numpy()

assert set(fold_ids) == {1, 2, 3}

print("Cross-validation folds loaded successfully.")


# ==========================================
# 3. PREPARE MULTILABEL TARGETS
# ==========================================

# Convert NumPy arrays to regular Python lists
# This fixes the JSON serialization error.

labels = [
    list(label)
    for label in train["risk_domains"]
]

mlb = MultiLabelBinarizer()

y = mlb.fit_transform(labels)

classes = list(mlb.classes_)

no_harm_idx = classes.index(
    "No Material Harm"
)

print("Number of harm categories:", len(classes))
print("Label matrix shape:", y.shape)


# ==========================================
# 4. PREPARE THREE INPUT REPRESENTATIONS
# ==========================================

context = (
    train["formatted_interaction"]
    .fillna("")
    .astype(str)
)

actions = (
    train["proposed_action"]
    .fillna("")
    .astype(str)
)

input_versions = {

    # Model A: Original full interaction
    "full_interaction": context,

    # Model B: Proposed action only
    "action_only": actions,

    # Model C: Full interaction with action emphasized
    "action_emphasized": (
        context
        + "\n\nACTION OF INTEREST:\n"
        + actions
        + "\n\nACTION OF INTEREST:\n"
        + actions
    )
}


# ==========================================
# 5. DECODE MULTILABEL PREDICTIONS
# ==========================================

def decode(probabilities):
    """
    Convert predicted label probabilities into
    one or two harm-domain labels.

    No Material Harm must appear alone.
    """

    ranked = np.argsort(-probabilities)

    selected = [
        int(i)
        for i in ranked
        if probabilities[i] >= 0.5
    ][:2]

    # Always predict at least one label
    if not selected:
        selected = [int(ranked[0])]

    # Enforce No Material Harm exclusivity
    if no_harm_idx in selected:

        if selected[0] == no_harm_idx:
            selected = [no_harm_idx]

        else:
            selected.remove(no_harm_idx)

    return [
        classes[i]
        for i in selected
    ]


# ==========================================
# 6. JACCARD SIMILARITY
# ==========================================

def jaccard(actual, predicted):
    """
    Jaccard = Intersection / Union
    """

    actual_set = set(actual)
    predicted_set = set(predicted)

    union = actual_set | predicted_set

    if not union:
        return 1.0

    return len(actual_set & predicted_set) / len(union)


# ==========================================
# 7. EXPERIMENT LOOP
# ==========================================

summaries = []
all_predictions = []

for version_name, texts in input_versions.items():

    print(
        f"\n========== {version_name} ==========",
        flush=True
    )

    scores = []
    missed_harm = 0
    false_alarms = 0
    exact_matches = 0

    for fold in [1, 2, 3]:

        # Training and validation indices
        tr = np.where(fold_ids != fold)[0]
        va = np.where(fold_ids == fold)[0]

        # ----------------------------------
        # TF-IDF FEATURES
        # ----------------------------------

        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        X_tr = vectorizer.fit_transform(
            texts.iloc[tr]
        )

        X_va = vectorizer.transform(
            texts.iloc[va]
        )

        # ----------------------------------
        # TRAIN CLASSIFIER
        # ----------------------------------

        classifier = OneVsRestClassifier(
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            )
        )

        classifier.fit(
            X_tr,
            y[tr]
        )

        # ----------------------------------
        # GENERATE PREDICTIONS
        # ----------------------------------

        probabilities = classifier.predict_proba(
            X_va
        )

        fold_scores = []

        for idx, prob in zip(va, probabilities):

            actual = labels[idx]

            predicted = decode(prob)

            score = jaccard(
                actual,
                predicted
            )

            fold_scores.append(score)
            scores.append(score)

            # Count harmful examples
            # incorrectly predicted harmless
            if (
                "No Material Harm" in predicted
                and "No Material Harm" not in actual
            ):
                missed_harm += 1

            # Count harmless examples
            # incorrectly predicted harmful
            if (
                "No Material Harm" in actual
                and "No Material Harm" not in predicted
            ):
                false_alarms += 1

            # Exact-match accuracy
            if set(actual) == set(predicted):
                exact_matches += 1

            # Save individual predictions
            all_predictions.append({

                "model": version_name,

                "example_id": str(
                    train.iloc[idx]["example_id"]
                ),

                "fold": int(fold),

                # Convert to regular lists
                # before JSON serialization
                "actual": json.dumps(
                    list(actual)
                ),

                "predicted": json.dumps(
                    list(predicted)
                ),

                "jaccard": float(score)
            })

        print(
            f"Fold {fold} Jaccard:",
            round(np.mean(fold_scores), 4),
            flush=True
        )

    # ----------------------------------
    # FINAL METRICS FOR THIS MODEL
    # ----------------------------------

    result = {

        "model": version_name,

        "oof_jaccard": round(
            float(np.mean(scores)), 4
        ),

        "exact_match_accuracy": round(
            exact_matches / len(labels), 4
        ),

        "missed_harm": missed_harm,

        "false_alarms": false_alarms
    }

    summaries.append(result)


# ==========================================
# 8. DISPLAY FINAL COMPARISON
# ==========================================

summary = pd.DataFrame(summaries)

print(
    "\n========== FINAL COMPARISON =========="
)

print(
    summary.to_string(index=False)
)


# ==========================================
# 9. SAVE RESULTS
# ==========================================

output = Path("subtask2/results")

output.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    output / "subtask2a_input_comparison.csv",
    index=False
)

predictions_df = pd.DataFrame(all_predictions)

predictions_df.to_csv(
    output / "subtask2a_input_comparison_predictions.csv",
    index=False
)

# Verify 568 predictions for each model
for model_name in input_versions:

    model_predictions = predictions_df[
        predictions_df["model"] == model_name
    ]

    assert len(model_predictions) == len(train)

    assert model_predictions["example_id"].is_unique


print("\n===================================")
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("===================================")

print("\nSaved files:")
print("- subtask2a_input_comparison.csv")
print("- subtask2a_input_comparison_predictions.csv")
