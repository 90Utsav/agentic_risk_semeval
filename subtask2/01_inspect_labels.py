
from datasets import load_dataset
from collections import Counter
from pathlib import Path
import pandas as pd

# Step 1: Load the official dataset
dataset = load_dataset(
    "AgentRisk/agentrisk-data-2027"
)

train = dataset["train"]

print("Training examples:", len(train))
print("Available columns:", train.column_names)

# Step 2: Extract harm-domain labels
all_labels = train["risk_domains"]

# Step 3: Count label frequencies
label_counts = Counter(
    label
    for labels in all_labels
    for label in labels
)

print("\n=== HARM DOMAIN DISTRIBUTION ===")

for label, count in label_counts.most_common():
    print(f"{label}: {count}")

# Step 4: Count how many labels each example has
labels_per_example = Counter(
    len(labels) for labels in all_labels
)

print("\n=== LABELS PER EXAMPLE ===")
print(dict(sorted(labels_per_example.items())))

# Step 5: Show three examples
print("\n=== SAMPLE EXAMPLES ===")

for example in train.select(range(3)):
    print("\nAction:", example["proposed_action"])
    print("Harm domains:", example["risk_domains"])

# Step 6: Save the distribution
output_dir = Path("subtask2/results")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(
    label_counts.items(),
    columns=["harm_domain", "frequency"]
).sort_values("frequency", ascending=False)

df.to_csv(
    output_dir / "harm_domain_distribution.csv",
    index=False
)

print("\nLabel distribution saved!")
