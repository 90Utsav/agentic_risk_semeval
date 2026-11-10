
"""
AgentRisk SemEval 2027
Subtask 2a — Experiment 2A-05

Compare the errors of three input representations:

1. Full interaction
2. Action only
3. Action emphasized

Metrics:
- Mean Jaccard similarity
- Exact-match accuracy
- Per-label precision
- Per-label recall
- Per-label F1 score
- Missed harmful actions
- False alarms

Uses previously saved out-of-fold predictions.
No model retraining is necessary.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
)


# ============================================
# 1. CONFIGURATION
# ============================================

RESULTS_DIR = Path("subtask2/results")

INPUT_FILE = (
    RESULTS_DIR
    / "subtask2a_input_comparison_predictions.csv"
)

MODEL_NAMES = [
    "full_interaction",
    "action_only",
    "action_emphasized"
]

NO_HARM = "No Material Harm"


# ============================================
# 2. LOAD PREVIOUS PREDICTIONS
# ============================================

print("\nLoading saved predictions...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing predictions file: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "model",
    "example_id",
    "fold",
    "actual",
    "predicted",
    "jaccard"
}

assert required_columns.issubset(df.columns), (
    "The prediction file is missing required columns."
)

# Convert JSON strings to regular Python lists
df["actual"] = df["actual"].apply(json.loads)
df["predicted"] = df["predicted"].apply(json.loads)

print("Total prediction rows:", len(df))
print("Models:", df["model"].unique().tolist())


# ============================================
# 3. VERIFY EXPERIMENT CONSISTENCY
# ============================================

print("\nVerifying experiment consistency...")

assert set(df["model"].unique()) == set(MODEL_NAMES)

reference = (
    df[df["model"] == "full_interaction"]
    .set_index("example_id")
    .sort_index()
)

assert len(reference) == 568
assert reference.index.is_unique

for model_name in MODEL_NAMES:

    model_df = (
        df[df["model"] == model_name]
        .set_index("example_id")
        .sort_index()
    )

    assert len(model_df) == 568
    assert model_df.index.is_unique

    # Ensure each model evaluated the same examples
    assert model_df.index.equals(reference.index)

    # Ensure actual labels and folds match
    for example_id in reference.index:

        assert (
            set(model_df.loc[example_id, "actual"])
            == set(reference.loc[example_id, "actual"])
        )

        assert (
            model_df.loc[example_id, "fold"]
            == reference.loc[example_id, "fold"]
        )

print("All models evaluated on matching examples!")


# ============================================
# 4. PREPARE MULTILABEL ENCODING
# ============================================

mlb = MultiLabelBinarizer()

mlb.fit(reference["actual"].tolist())

print("\nHarm categories:", len(mlb.classes_))


# ============================================
# 5. EVALUATE EACH MODEL
# ============================================

all_reports = []
overall_results = []

for model_name in MODEL_NAMES:

    print(f"\n========== {model_name} ==========")

    model_df = df[
        df["model"] == model_name
    ].copy()

    actual = model_df["actual"].tolist()
    predicted = model_df["predicted"].tolist()

    y_true = mlb.transform(actual)
    y_pred = mlb.transform(predicted)

    # Per-label precision, recall and F1
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0
        )
    )

    exact_match = accuracy_score(
        y_true,
        y_pred
    )

    # Calculate missed harmful examples
    missed_harm = sum(
        NO_HARM in p and NO_HARM not in a
        for a, p in zip(actual, predicted)
    )

    # Calculate false alarms
    false_alarms = sum(
        NO_HARM in a and NO_HARM not in p
        for a, p in zip(actual, predicted)
    )

    # Overall model results
    overall_results.append({
        "model": model_name,
        "jaccard": model_df["jaccard"].mean(),
        "exact_match": exact_match,
        "missed_harm": missed_harm,
        "false_alarms": false_alarms
    })

    # Store metrics for every label
    for i, label in enumerate(mlb.classes_):

        all_reports.append({
            "model": model_name,
            "label": label,
            "support": int(support[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i])
        })

    print(
        "Jaccard:",
        round(model_df["jaccard"].mean(), 4)
    )

    print(
        "Exact-match accuracy:",
        round(exact_match, 4)
    )

    print("Missed harms:", missed_harm)
    print("False alarms:", false_alarms)


# ============================================
# 6. COMPARE OVERALL PERFORMANCE
# ============================================

overall_df = pd.DataFrame(overall_results)

print("\n========== OVERALL COMPARISON ==========")

print(
    overall_df.round(4).to_string(index=False)
)


# ============================================
# 7. COMPARE PER-LABEL RECALL
# ============================================

report_df = pd.DataFrame(all_reports)

recall_table = report_df.pivot(
    index="label",
    columns="model",
    values="recall"
)

# Include the number of real examples per label
support_table = (
    report_df[report_df["model"] == "full_interaction"]
    .set_index("label")["support"]
)

recall_table.insert(
    0,
    "support",
    support_table
)

# Measure improvement compared to baseline
recall_table["action_only_gain"] = (
    recall_table["action_only"]
    - recall_table["full_interaction"]
)

recall_table["emphasized_gain"] = (
    recall_table["action_emphasized"]
    - recall_table["full_interaction"]
)

recall_table = recall_table.sort_values(
    "support",
    ascending=False
)

print("\n========== RECALL BY HARM DOMAIN ==========")

print(
    recall_table.round(4).to_string()
)


# ============================================
# 8. COMPARE PER-LABEL F1
# ============================================

f1_table = report_df.pivot(
    index="label",
    columns="model",
    values="f1_score"
)

print("\n========== F1 BY HARM DOMAIN ==========")

print(
    f1_table.round(4).to_string()
)


# ============================================
# 9. COMPARE INDIVIDUAL EXAMPLES
# ============================================

# Create one column of Jaccard scores per model
example_comparison = df.pivot(
    index="example_id",
    columns="model",
    values="jaccard"
)

example_comparison["emphasized_gain"] = (
    example_comparison["action_emphasized"]
    - example_comparison["full_interaction"]
)

example_comparison["action_only_gain"] = (
    example_comparison["action_only"]
    - example_comparison["full_interaction"]
)

print("\n========== EXAMPLE-LEVEL COMPARISON ==========")

for model_name in ["action_only", "action_emphasized"]:

    difference = (
        example_comparison[model_name]
        - example_comparison["full_interaction"]
    )

    print(f"\n{model_name}")
    print("Improved examples:", int((difference > 0).sum()))
    print("Worsened examples:", int((difference < 0).sum()))
    print("Unchanged examples:", int((difference == 0).sum()))


# ============================================
# 10. SAVE RESULTS
# ============================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

overall_df.to_csv(
    RESULTS_DIR / "subtask2a_overall_comparison.csv",
    index=False
)

report_df.to_csv(
    RESULTS_DIR / "subtask2a_detailed_label_metrics.csv",
    index=False
)

recall_table.to_csv(
    RESULTS_DIR / "subtask2a_recall_comparison.csv"
)

f1_table.to_csv(
    RESULTS_DIR / "subtask2a_f1_comparison.csv"
)

example_comparison.to_csv(
    RESULTS_DIR / "subtask2a_example_comparison.csv"
)

print("\n========================================")
print("EXPERIMENT 2A-05 COMPLETED SUCCESSFULLY")
print("========================================")

print("\nSaved files:")
print("- subtask2a_overall_comparison.csv")
print("- subtask2a_detailed_label_metrics.csv")
print("- subtask2a_recall_comparison.csv")
print("- subtask2a_f1_comparison.csv")
print("- subtask2a_example_comparison.csv")
