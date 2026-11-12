"""AgentRisk SemEval 2027 — Subtask 2b ModernBERT grouped-CV experiment.

Run from the repository root:
    python -u subtask2/10_train_modernbert_drivers.py --smoke
    python -u subtask2/10_train_modernbert_drivers.py

Requires torch, transformers, datasets, scikit-learn, pandas, numpy.
"""

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
FOLDS_FILE = ROOT / "subtask1/configs/cv_folds.csv"
RESULTS_DIR = ROOT / "subtask2/results"
MODEL_NAME = "answerdotai/ModernBERT-base"
NO_RISK = "No Material Risk"
SEED = 42


class EncodedDataset(Dataset):
    def __init__(self, encoded, labels, indices):
        self.encoded = encoded
        self.labels = labels
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        return {
            "input_ids": self.encoded["input_ids"][i],
            "attention_mask": self.encoded["attention_mask"][i],
            "labels": self.labels[i],
        }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def decode_probabilities(probabilities, names):
    """Match the previously established 0.5-threshold / top-2 rule."""
    names = list(names)
    no_risk_idx = names.index(NO_RISK)
    decoded = []
    for row in probabilities:
        selected = np.flatnonzero(row >= 0.5).tolist()
        if not selected:
            selected = [int(np.argmax(row))]
        if no_risk_idx in selected and len(selected) > 1:
            others = [i for i in selected if i != no_risk_idx]
            strongest_other = max(others, key=lambda i: row[i])
            if row[no_risk_idx] >= row[strongest_other]:
                selected = [no_risk_idx]
            else:
                selected = others
        selected = sorted(selected, key=lambda i: row[i], reverse=True)[:2]
        decoded.append([names[i] for i in selected])
    return decoded


def jaccard(actual, predicted):
    actual, predicted = set(actual), set(predicted)
    union = actual | predicted
    return len(actual & predicted) / len(union) if union else 1.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="24 training / 8 validation examples, one epoch, fold 1. Not a valid benchmark.")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), default=None, help="Run just one fold for debugging.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.max_length < 32 or args.batch_size < 1 or args.grad_accum < 1 or args.epochs < 1:
        parser.error("Invalid sequence length, batch size, accumulation steps, or epochs")
    return args


def evaluate_model(model, loader, device):
    model.eval()
    parts = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            logits = model(**inputs).logits
            parts.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(parts, axis=0)


def main():
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this Python environment. Check torch.backends.mps.is_available().")
    device = torch.device("mps")
    seed_everything(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print(f"Checkpoint: {MODEL_NAME}")
    print("Mode:", "SMOKE TEST (not a benchmark)" if args.smoke else "Grouped cross-validation")
    print("Loading official AgentRisk training data...")
    df = load_dataset("AgentRisk/agentrisk-data-2027")["train"].to_pandas()
    if not df["example_id"].is_unique:
        raise ValueError("Dataset has duplicate example IDs")
    folds_df = pd.read_csv(FOLDS_FILE)
    if not folds_df["example_id"].is_unique:
        raise ValueError("Fold assignments have duplicate example IDs")
    fold_by_id = folds_df.set_index("example_id")["validation_fold"]
    df["fold"] = df["example_id"].map(fold_by_id)
    if df["fold"].isna().any() or set(df["fold"].astype(int).unique()) != {1, 2, 3}:
        raise ValueError("Fold assignments missing or unexpected; expected folds 1,2,3")

    labels = [list(value) for value in df["risk_drivers"]]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    names = list(mlb.classes_)
    assert len(names) == 7 and NO_RISK in names
    label_tensors = torch.tensor(y, dtype=torch.float32)

    # Action comes BEFORE context: right-side truncation will not remove it.
    actions = df["proposed_action"].fillna("").astype(str).tolist()
    contexts = df["formatted_interaction"].fillna("").astype(str).tolist()
    texts = [
        f"PROPOSED ACTION:\n{a}\n\nPROPOSED ACTION:\n{a}\n\nAGENT INTERACTION:\n{c}"
        for a, c in zip(actions, contexts)
    ]
    print("Labels:", names)
    print("Loading tokenizer and encoding inputs...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lengths = [len(ids) for ids in tokenizer(texts, padding=False, truncation=False)["input_ids"]]
    n_truncated = sum(length > args.max_length for length in lengths)
    print(f"Inputs longer than {args.max_length} tokens: {n_truncated}/{len(df)}")
    encoded = tokenizer(
        texts,
        padding="max_length",  # Constant tensor shapes reduce MPS compilation/cache growth.
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    chosen_folds = [1] if args.smoke else ([args.fold] if args.fold else [1, 2, 3])
    all_records = []
    for fold in chosen_folds:
        seed_everything(SEED + fold)
        train_idx = np.flatnonzero(df["fold"].to_numpy() != fold)
        val_idx = np.flatnonzero(df["fold"].to_numpy() == fold)
        if args.smoke:
            train_idx = train_idx[:24]
            val_idx = val_idx[:8]
        print(f"\n===== FOLD {fold}: training={len(train_idx)}, held-out={len(val_idx)} =====", flush=True)
        train_loader = DataLoader(
            EncodedDataset(encoded, label_tensors, train_idx),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            EncodedDataset(encoded, label_tensors, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        print("Loading pretrained model; classification head starts randomly initialized...", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(names),
            id2label={i: name for i, name in enumerate(names)},
            label2id={name: i for i, name in enumerate(names)},
            problem_type="multi_label_classification",
            attn_implementation="sdpa",
        )
        if not args.no_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        epochs = 1 if args.smoke else args.epochs
        steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
        total_steps = steps_per_epoch * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.1 * total_steps)),
            num_training_steps=total_steps,
        )
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                loss = criterion(logits.float(), batch["labels"].to(device))
                running_loss += loss.detach().item()
                remaining_in_group = len(train_loader) - (step // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, remaining_in_group)
                (loss / group_size).backward()
                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            print(f"Fold {fold} epoch {epoch + 1}/{epochs}: training loss={running_loss / len(train_loader):.4f}", flush=True)

        probabilities = evaluate_model(model, val_loader, device)
        predictions = decode_probabilities(probabilities, names)
        actual = [labels[i] for i in val_idx]
        scores = [jaccard(a, p) for a, p in zip(actual, predictions)]
        print(f"Fold {fold} held-out Jaccard: {np.mean(scores):.4f}", flush=True)
        for pos, idx in enumerate(val_idx):
            all_records.append({
                "example_id": df.iloc[int(idx)]["example_id"],
                "fold": int(fold),
                "actual": json.dumps(actual[pos]),
                "predicted": json.dumps(predictions[pos]),
                "jaccard": float(scores[pos]),
                "probabilities": json.dumps(probabilities[pos].astype(float).tolist()),
            })
        if not args.smoke:
            pd.DataFrame(all_records).to_csv(
                RESULTS_DIR / "subtask2b_modernbert_partial_oof.csv", index=False
            )
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        torch.mps.empty_cache()

    if args.smoke:
        print("\nSMOKE TEST PASSED. These 8 validation predictions are NOT a performance benchmark.")
        return

    records_df = pd.DataFrame(all_records)
    if len(chosen_folds) == 3:
        if len(records_df) != len(df) or not records_df["example_id"].is_unique:
            raise ValueError("OOF predictions must contain exactly one row per example")
        output_name = "subtask2b_modernbert_oof.csv"
    else:
        output_name = f"subtask2b_modernbert_fold{chosen_folds[0]}_oof.csv"
    records_df.to_csv(RESULTS_DIR / output_name, index=False)
    actual = [json.loads(s) for s in records_df["actual"]]
    predicted = [json.loads(s) for s in records_df["predicted"]]
    yt, yp = mlb.transform(actual), mlb.transform(predicted)
    prec, rec, f1, support = precision_recall_fscore_support(yt, yp, average=None, zero_division=0)
    metric_df = pd.DataFrame({
        "label": names, "support": support,
        "precision": prec, "recall": rec, "f1": f1,
    })
    missed = sum(NO_RISK in p and NO_RISK not in a for a, p in zip(actual, predicted))
    false_alarms = sum(NO_RISK in a and NO_RISK not in p for a, p in zip(actual, predicted))
    print("\n===== EVALUATION =====")
    print("Fold coverage:", chosen_folds)
    print("Mean Jaccard:", round(float(records_df["jaccard"].mean()), 4))
    print("Exact-match accuracy:", round(float(accuracy_score(yt, yp)), 4))
    print("Missed risks:", missed)
    print("False alarms:", false_alarms)
    print("\nPer-label metrics:\n", metric_df.round(4).to_string(index=False))
    if len(chosen_folds) == 3:
        metric_df.to_csv(RESULTS_DIR / "subtask2b_modernbert_per_label.csv", index=False)
        config = {
            "checkpoint": MODEL_NAME,
            "python": os.sys.version,
            "torch": torch.__version__,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "input_representation": "proposed action twice, before full interaction",
            "threshold": 0.5,
            "max_predicted_labels": 2,
            "seed": SEED,
            "folds": [1, 2, 3],
            "labels": names,
            "validation_used_for_model_selection": False,
        }
        (RESULTS_DIR / "subtask2b_modernbert_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        print("\nSaved:", RESULTS_DIR / output_name)
        print("Saved: subtask2b_modernbert_per_label.csv")
        print("Saved: subtask2b_modernbert_config.json")
    else:
        print("\nPARTIAL FOLD ONLY: do not compare this score to full OOF 0.6866")
        print("Saved:", RESULTS_DIR / output_name)


if __name__ == "__main__":
    main()
