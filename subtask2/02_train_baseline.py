
import json
import numpy as np
import pandas as pd

from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# 1. Load training examples
ds = load_dataset("AgentRisk/agentrisk-data-2027")
train = ds["train"].to_pandas()

X = train["formatted_interaction"].fillna("").astype(str)
labels = train["risk_domains"].tolist()

# 2. Convert label sets to binary vectors
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)

print("Text examples:", len(X))
print("Label matrix shape:", y.shape)

# 3. Reuse saved fold assignments if available
fold_file = Path("subtask1/configs/cv_folds.csv")

if fold_file.exists():
    saved = pd.read_csv(fold_file)
    assert saved["example_id"].is_unique
    fold_mapping = saved.set_index("example_id")["validation_fold"]
    fold_ids = train["example_id"].map(fold_mapping)
    assert fold_ids.notna().all()
    fold_ids = fold_ids.astype(int).to_numpy()

    assert set(fold_ids) == {1, 2, 3}

else:
    # Regenerate folds using official scenario groups
    manifest_path = hf_hub_download(
        repo_id="AgentRisk/agentrisk-data-2027",
        filename="metadata/split_manifest.jsonl",
        repo_type="dataset"
    )
    manifest = pd.read_json(manifest_path, lines=True)
    groups = train["example_id"].map(
        manifest.set_index("example_id")["scenario_group"]
    )
    assert groups.notna().all()

    cv = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=42
    )
    fold_ids = np.zeros(len(train), dtype=int)
    for fold_num, (_, val_idx) in enumerate(
        cv.split(X, train["risk_severity"], groups),
        start=1
    ):
        fold_ids[val_idx] = fold_num

# 4. Predict a legal set of 1–2 domains
classes = list(mlb.classes_)
no_harm = classes.index("No Material Harm")

def decode_prediction(probabilities):
    ranked = np.argsort(-probabilities)

    selected = [
        int(i) for i in ranked
        if probabilities[i] >= 0.5
    ][:2]

    if not selected:
        selected = [int(ranked[0])]

    # No Material Harm must be alone
    if no_harm in selected:
        if selected[0] == no_harm:
            selected = [no_harm]
        else:
            selected.remove(no_harm)

    return [classes[i] for i in selected]

def jaccard(actual, predicted):
    a, p = set(actual), set(predicted)
    return len(a & p) / len(a | p)

# 5. Three-fold grouped cross-validation
records = []

for fold in [1, 2, 3]:
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]

    # Fit text features ONLY on fold-training data
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_tr = vectorizer.fit_transform(X.iloc[tr_idx])
    X_va = vectorizer.transform(X.iloc[va_idx])

    # One binary Logistic Regression per label
    classifier = OneVsRestClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    )

    classifier.fit(X_tr, y[tr_idx])

    probabilities = classifier.predict_proba(X_va)

    fold_scores = []

    for row_idx, prob in zip(va_idx, probabilities):
        actual = list(labels[row_idx])
        predicted = decode_prediction(prob)
        score = jaccard(actual, predicted)

        fold_scores.append(score)

        records.append({
            "example_id": train.iloc[row_idx]["example_id"],
            "fold": fold,
            "actual": json.dumps(actual),
            "predicted": json.dumps(predicted),
            "jaccard": score
        })

    print(
        f"Fold {fold} Jaccard:",
        round(np.mean(fold_scores), 4)
    )

# 6. Save predictions and results
results = pd.DataFrame(records)

assert len(results) == 568
assert results["example_id"].is_unique

output = Path("subtask2/results")
output.mkdir(parents=True, exist_ok=True)

results.to_csv(
    output / "subtask2a_baseline_cv.csv",
    index=False
)

print("\nOverall OOF Jaccard:",
      round(results["jaccard"].mean(), 4))

print("Majority reference Jaccard:",
      round(sum("No Material Harm" in x for x in labels) / len(labels), 4))

print("Saved:", output / "subtask2a_baseline_cv.csv")
