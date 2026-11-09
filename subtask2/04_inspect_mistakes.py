
import json
import pandas as pd
from datasets import load_dataset

# Load predictions from our baseline
predictions = pd.read_csv(
    "subtask2/results/subtask2a_baseline_cv.csv"
)

# Load original training examples
dataset = load_dataset("AgentRisk/agentrisk-data-2027")
train = dataset["train"].to_pandas()

# Combine predictions with their original interactions
df = predictions.merge(
    train[[
        "example_id",
        "formatted_interaction",
        "proposed_action"
    ]],
    on="example_id",
    how="left",
    validate="one_to_one"
)

assert df["proposed_action"].notna().all()

# Parse saved label lists
df["actual_labels"] = df["actual"].apply(json.loads)
df["predicted_labels"] = df["predicted"].apply(json.loads)

# Find harmful cases predicted as harmless
missed = df[
    df.apply(
        lambda row:
            "No Material Harm" in row["predicted_labels"]
            and "No Material Harm" not in row["actual_labels"],
        axis=1
    )
].copy()

print("Total missed harmful examples:", len(missed))

# Inspect eight examples
for _, row in missed.head(8).iterrows():

    print("\n" + "=" * 75)
    print("EXAMPLE:", row["example_id"])

    print("\nACTUAL LABELS:")
    print(row["actual_labels"])

    print("\nMODEL PREDICTION:")
    print(row["predicted_labels"])

    print("\nPROPOSED ACTION:")
    print(str(row["proposed_action"])[:1500])

    print("\nEND OF INTERACTION HISTORY:")
    print(str(row["formatted_interaction"])[-1000:])

# Save full examples for later manual review
missed.to_csv(
    "subtask2/results/subtask2a_missed_harm_detailed.csv",
    index=False
)

print("\nDetailed mistakes saved successfully!")
