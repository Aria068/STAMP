"""
implicit_classify.py — Implicit attribute leakage classifier (Table 5).

Trains a BERT classifier on clean SynthPAI text to predict implicit attributes
(gender_id, age_id, education_id), then evaluates whether synthetic rewrites
still leak those attributes. Lower F1 on synthetic = better privacy.

Usage:
    # Evaluate all three attributes at once
    python implicit_classify.py \\
        --raw_train  clean_data/SynthPAI_AuthorInfo_train.csv \\
        --raw_test   clean_data/SynthPAI_AuthorInfo_test.csv \\
        --syn_path   results/ours_synthpai_syn.csv \\
        --attr_fields gender_id age_id education_id

    # Single attribute
    python implicit_classify.py \\
        --raw_train  clean_data/SynthPAI_AuthorInfo_train.csv \\
        --raw_test   clean_data/SynthPAI_AuthorInfo_test.csv \\
        --syn_path   results/dpmlm_SynthPAI_AuthorInfo_test.csv \\
        --attr_fields gender_id
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn import metrics


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "Accuracy": metrics.accuracy_score(labels, predictions),
        "F1 Score": metrics.f1_score(labels, predictions, average="macro"),
        "MCC": metrics.matthews_corrcoef(labels, predictions),
    }


def make_hf_dataset(tokenizer, texts, labels, max_length):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": np.array(labels),
    })


def train_and_eval(
    train_texts, val_texts, test_texts,
    train_labels, val_labels, test_labels,
    num_labels, attr_field, tokenizer, max_length, epochs,
):
    """Train BERT on clean data; evaluate on synthetic (test) texts."""
    train_ds = make_hf_dataset(tokenizer, train_texts, train_labels, max_length)
    val_ds = make_hf_dataset(tokenizer, val_texts, val_labels, max_length)
    test_ds = make_hf_dataset(tokenizer, test_texts, test_labels, max_length)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=num_labels
    )
    training_args = TrainingArguments(
        output_dir=f"./output_{attr_field}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="F1 Score",
        save_total_limit=1,
        report_to="none",
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    results = trainer.evaluate(eval_dataset=test_ds)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Implicit attribute leakage evaluation on SynthPAI (Table 5)."
    )
    parser.add_argument(
        "--raw_train", default="clean_data/SynthPAI_AuthorInfo_train.csv",
        help="Clean SynthPAI train CSV (used to train the attribute classifier).",
    )
    parser.add_argument(
        "--raw_test", default="clean_data/SynthPAI_AuthorInfo_test.csv",
        help="Clean SynthPAI test CSV (provides ground-truth labels for the test set).",
    )
    parser.add_argument(
        "--syn_path", required=True,
        help="CSV with synthetic (rewritten) SynthPAI test texts to evaluate.",
    )
    parser.add_argument(
        "--text_field", default="text",
        help="Column name for text in all CSVs (default: text).",
    )
    parser.add_argument(
        "--attr_fields", nargs="+", default=["gender_id"],
        help="Attribute columns to evaluate. Can pass multiple: gender_id age_id education_id.",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    clean_train = pd.read_csv(args.raw_train)
    clean_test = pd.read_csv(args.raw_test)
    syn_test = pd.read_csv(args.syn_path)

    # Align synthetic test with clean_test (row-for-row correspondence)
    min_len = min(len(clean_test), len(syn_test))
    clean_test = clean_test.iloc[:min_len].reset_index(drop=True)
    syn_test = syn_test.iloc[:min_len].reset_index(drop=True)

    idxs = np.arange(len(clean_train))
    train_idxs, val_idxs = train_test_split(idxs, test_size=0.10, random_state=42)
    print(f"Train: {len(train_idxs)}, Val: {len(val_idxs)}, Test: {min_len}")

    all_results = {}
    for attr in args.attr_fields:
        missing = [
            name for name, df in [
                ("raw_train", clean_train), ("raw_test", clean_test)
            ] if attr not in df.columns
        ]
        if missing:
            print(f"Skipping '{attr}': not found in {missing}")
            continue

        tr = clean_train.iloc[train_idxs][[args.text_field, attr]].dropna()
        va = clean_train.iloc[val_idxs][[args.text_field, attr]].dropna()

        # Test split: synthetic text + clean ground-truth labels
        te_text = syn_test[args.text_field].tolist()
        te_labels = clean_test[attr].tolist()
        # Drop rows where either is NaN
        te_pairs = [(t, l) for t, l in zip(te_text, te_labels)
                    if pd.notna(t) and pd.notna(l)]
        te_text, te_labels = zip(*te_pairs) if te_pairs else ([], [])

        # Re-encode labels to contiguous 0-indexed integers.
        # Needed when the raw label column has gaps (e.g. values 0-9 but only
        # 9 distinct classes, causing an out-of-range CUDA assert).
        unique_labels = sorted(clean_train[attr].dropna().unique())
        label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        num_labels = len(unique_labels)
        print(f"\n--- {attr} (num_labels={num_labels}, mapping={label2idx}) ---")

        tr_labels = [label2idx[l] for l in tr[attr].tolist()]
        va_labels = [label2idx[l] for l in va[attr].tolist()]
        te_pairs2 = [(t, label2idx[l]) for t, l in zip(te_text, te_labels) if l in label2idx]
        if not te_pairs2:
            print(f"Skipping '{attr}': no test labels overlap with train labels")
            continue
        te_text, te_labels = zip(*te_pairs2)
        te_text, te_labels = list(te_text), list(te_labels)

        results = train_and_eval(
            tr[args.text_field].tolist(), va[args.text_field].tolist(), te_text,
            tr_labels, va_labels, te_labels,
            num_labels, attr, tokenizer, args.max_length, args.epochs,
        )
        print(f"Results [{attr}]:", results)
        all_results[attr] = results

    print("\n=== Summary (Table 5) ===")
    for attr, res in all_results.items():
        f1 = res.get("eval_F1 Score", float("nan"))
        acc = res.get("eval_Accuracy", float("nan"))
        print(f"  {attr:20s}  F1={f1:.4f}  Acc={acc:.4f}")


if __name__ == "__main__":
    main()
