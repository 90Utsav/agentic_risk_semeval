
"""
AgentRisk SemEval 2027
Subtask 2b — Experiment 2B-01

Explore the risk-driver labels before model training.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# 1. Load official dataset

print("Loading AgentRisk dataset...")

dataset = load_dataset(
    "AgentRisk/agentrisk-data-2027"
)

train = dataset["train"]

print("\nTraining examples:", len(train))


# 2. Extract risk-driver labels

driver_lists = [
    list(labels) if labels is not None else []
    for labels in train["risk_drivers"]
]

driver_counts = Counter(
    driver
    for drivers in driver_lists
    for driver in drivers
)


# 3. Display driver frequencies

print("\n========== RISK DRIVER DISTRIBUTION ==========")

for driver, count in driver_counts.most_common():
    percentage = 100 * count / len(train)

    print(
        f"{driver:<45} "
        f"{count:>4}  ({percentage:.1f}%)"
    )


# 4. Analyze label counts per example

labels_per_example = Counter(
    len(drivers)
    for drivers in driver_lists
)

print("\n========== DRIVERS PER EXAMPLE ==========")

for count, frequency in sorted(
    labels_per_example.items()
):
    print(
        f"{count} drivers: {frequency} examples"
    )


# 5. Inspect example labels

print("\n========== EXAMPLE ANNOTATIONS ==========")

for i in range(min(5, len(train))):
    print(f"\nExample {i + 1}")
    print("ID:", train[i]["example_id"])
    print("Severity:", train[i]["risk_severity"])
    print("Harm domains:", train[i]["risk_domains"])
    print("Risk drivers:", train[i]["risk_drivers"])


# 6. Save distribution

output_dir = Path("subtask2/results")
output_dir.mkdir(parents=True, exist_ok=True)

distribution = pd.DataFrame(
    [
        {
            "risk_driver": driver,
            "frequency": count,
            "percentage": 100 * count / len(train)
        }
        for driver, count in driver_counts.most_common()
    ]
)

distribution.to_csv(
    output_dir / "subtask2b_driver_distribution.csv",
    index=False
)

print("\n========== SUMMARY ==========")
print("Unique risk drivers:", len(driver_counts))
print("Total driver annotations:", sum(driver_counts.values()))
print("Total training examples:", len(train))
print("\nSaved: subtask2/results/subtask2b_driver_distribution.csv")
