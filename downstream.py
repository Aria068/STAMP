"""
downstream.py — Downstream utility evaluation (Table 6).

Train → Test protocol: train a BERT classifier on SYNTHETIC rewrites
(labels transferred from the original), evaluate on the REAL held-out test
set. Higher F1/Accuracy = better utility preserved after rewriting.

Supported datasets (paper Table 6):
    AGNews   : clean_data/ag_news_sample_{train,test}.csv  label_field=label
    ECInstruct: clean_data/ECInstruct_{train,test}.csv     label_field=label
    TReview  : clean_data/trust_pilot_reviews_{train,eval}.csv  label_field=<check>

Usage:
    # AGNews
    python downstream.py \\
        --raw_train  clean_data/ag_news_sample_train.csv \\
        --raw_test   clean_data/ag_news_sample_test.csv \\
        --syn_train  results/ours_ag_news_sample_train_syn.csv \\
        --text_field review --label_field label

    # ECInstruct
    python downstream.py \\
        --raw_train  clean_data/ECInstruct_train.csv \\
        --raw_test   clean_data/ECInstruct_test.csv \\
        --syn_train  results/ours_ECInstruct_train_syn.csv \\
        --text_field review --label_field label

    # TReview (trust_pilot)
    python downstream.py \\
        --raw_train  clean_data/trust_pilot_reviews_train.csv \\
        --raw_test   clean_data/trust_pilot_reviews_eval.csv \\
        --syn_train  results/ours_trust_pilot_reviews_train_syn.csv \\
        --text_field review --label_field label
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
        "labels": np.array(labels, dtype=np.int64),
    })


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Downstream utility: train on synthetic rewrites, test on real data. "
            "Higher performance = better utility preserved."
        )
    )
    parser.add_argument(
        "--raw_train", required=True,
        help="CSV with original TRAINING texts and labels.",
    )
    parser.add_argument(
        "--raw_test", required=True,
        help="CSV with original TEST texts and labels (used as held-out test).",
    )
    parser.add_argument(
        "--syn_train", required=True,
        help=(
            "CSV with synthetic rewrites of the TRAINING set "
            "(row-aligned with raw_train; labels transferred from raw_train)."
        ),
    )
    parser.add_argument(
        "--text_field", default="review",
        help="Column name for text in all CSVs (default: review).",
    )
    parser.add_argument(
        "--label_field", default="label",
        help="Column name for label in all CSVs (default: label).",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    print(f"Raw train:  {args.raw_train}")
    print(f"Raw test:   {args.raw_test}")
    print(f"Syn train:  {args.syn_train}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    raw_train_df = pd.read_csv(args.raw_train)
    raw_test_df = pd.read_csv(args.raw_test)
    syn_train_df = pd.read_csv(args.syn_train)

    for col, df, path in [
        (args.text_field, raw_train_df, args.raw_train),
        (args.label_field, raw_train_df, args.raw_train),
        (args.text_field, raw_test_df, args.raw_test),
        (args.label_field, raw_test_df, args.raw_test),
        (args.text_field, syn_train_df, args.syn_train),
    ]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {path}. Available: {list(df.columns)}"
            )

    # Align synthetic train with raw_train (row-for-row)
    n_train = min(len(raw_train_df), len(syn_train_df))
    raw_train_df = raw_train_df.iloc[:n_train].reset_index(drop=True)
    syn_train_df = syn_train_df.iloc[:n_train].reset_index(drop=True)

    # Transfer labels from raw → synthetic
    syn_train_df = syn_train_df[[args.text_field]].copy()
    syn_train_df[args.label_field] = raw_train_df[args.label_field].values

    # Create a small validation split from synthetic training data
    idxs = np.arange(n_train)
    train_idxs, val_idxs = train_test_split(idxs, test_size=0.10, random_state=42)

    train_df = syn_train_df.iloc[train_idxs][[args.text_field, args.label_field]].dropna()
    val_df = syn_train_df.iloc[val_idxs][[args.text_field, args.label_field]].dropna()
    # Test on REAL (unseen) data
    test_df = raw_test_df[[args.text_field, args.label_field]].dropna()

    # Re-encode labels to contiguous 0-indexed integers.
    # Handles string labels (e.g. "positive"/"negative") and non-contiguous ints.
    unique_labels = sorted(raw_train_df[args.label_field].dropna().unique(), key=str)
    label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Label mapping: {label2idx}")
    print(
        f"Classes: {num_classes} | "
        f"Train (syn): {len(train_df)} | Val (syn): {len(val_df)} | Test (real): {len(test_df)}"
    )

    train_ds = make_hf_dataset(
        tokenizer, train_df[args.text_field].tolist(),
        [label2idx[l] for l in train_df[args.label_field].tolist()], args.max_length,
    )
    val_ds = make_hf_dataset(
        tokenizer, val_df[args.text_field].tolist(),
        [label2idx[l] for l in val_df[args.label_field].tolist()], args.max_length,
    )
    test_ds = make_hf_dataset(
        tokenizer, test_df[args.text_field].tolist(),
        [label2idx[l] for l in test_df[args.label_field].tolist()], args.max_length,
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=num_classes
    )
    training_args = TrainingArguments(
        output_dir="./output_downstream",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
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

    test_results = trainer.evaluate(eval_dataset=test_ds)
    print("\n=== Downstream Utility Results (Table 6) ===")
    print(test_results)


if __name__ == "__main__":
    main()
