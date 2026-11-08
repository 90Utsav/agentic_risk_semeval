
import json
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report
)

# ==========================================
# 1. LOAD SAVED PREDICTIONS
# ==========================================

print("Starting Subtask 2a Error Analysis...")

input_file = Path("subtask2/results/subtask2a_baseline_cv.csv")

if not input_file.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {input_file}"
    )

df = pd.read_csv(input_file)

print("\nTotal predictions:", len(df))

# Convert JSON strings back into Python lists
actual = df["actual"].apply(json.loads).tolist()
predicted = df["predicted"].apply(json.loads).tolist()


# ==========================================
# 2. CONVERT LABELS INTO BINARY VECTORS
# ==========================================

mlb = MultiLabelBinarizer()

y_true = mlb.fit_transform(actual)
y_pred = mlb.transform(predicted)

print("\nNumber of harm categories:", len(mlb.classes_))
print("Label categories:", list(mlb.classes_))


# ==========================================
# 3. CALCULATE PRECISION, RECALL AND F1
# ==========================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )
)

report = pd.DataFrame({
    "label": mlb.classes_,
    "support": support,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
})

report = report.sort_values(
    "support",
    ascending=False
)

print("\n========== PERFORMANCE BY HARM DOMAIN ==========")

print(report.to_string(index=False))


# ==========================================
# 4. CALCULATE OVERALL METRICS
# ==========================================

exact_match = accuracy_score(y_true, y_pred)

print("\n========== OVERALL METRICS ==========")

print("Mean Jaccard:", round(df["jaccard"].mean(), 4))

print("Exact-match accuracy:", round(exact_match, 4))


# ==========================================
# 5. ANALYZE MISSED HARMFUL ACTIONS
# ==========================================

no_harm = "No Material Harm"

missed_harm = df[
    [
        no_harm in p and no_harm not in a
        for a, p in zip(actual, predicted)
    ]
]

print("\n========== MISSED HARMFUL ACTIONS ==========")

print(
    "Harmful examples predicted as No Material Harm:",
    len(missed_harm)
)

print("\nFirst 10 missed harmful examples:")

print(
    missed_harm.head(10).to_string(index=False)
)


# ==========================================
# 6. ANALYZE FALSE ALARMS
# ==========================================

false_alarms = df[
    [
        no_harm in a and no_harm not in p
        for a, p in zip(actual, predicted)
    ]
]

print("\n========== FALSE ALARMS ==========")

print(
    "Harmless examples predicted as harmful:",
    len(false_alarms)
)


# ==========================================
# 7. ANALYZE THE WORST PREDICTIONS
# ==========================================

worst_predictions = df.sort_values(
    "jaccard",
    ascending=True
)

print("\n========== 10 WORST PREDICTIONS ==========")

print(
    worst_predictions.head(10).to_string(index=False)
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

output_dir = Path("subtask2/results")
output_dir.mkdir(parents=True, exist_ok=True)

report.to_csv(
    output_dir / "subtask2a_per_label_metrics.csv",
    index=False
)

missed_harm.to_csv(
    output_dir / "subtask2a_missed_harm.csv",
    index=False
)

false_alarms.to_csv(
    output_dir / "subtask2a_false_alarms.csv",
    index=False
)

worst_predictions.head(30).to_csv(
    output_dir / "subtask2a_worst_predictions.csv",
    index=False
)

print("\n===================================")
print("ERROR ANALYSIS COMPLETED")
print("===================================")

print("\nSaved files:")
print("- subtask2a_per_label_metrics.csv")
print("- subtask2a_missed_harm.csv")
print("- subtask2a_false_alarms.csv")
print("- subtask2a_worst_predictions.csv")
