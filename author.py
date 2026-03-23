import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn import metrics

from typing import Dict

Stats = Dict[str, Dict[str, float]]
static_trainers = {}


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
max_length = 512

author_id_field = "author_id"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = metrics.accuracy_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions, average="macro")
    mcc = metrics.matthews_corrcoef(labels, predictions)

    stats = {"Accuracy": acc, "F1 Score": f1, "Mathew correleation coefficient": mcc}
    return stats


def get_trained_trainer(train_dataset, val_dataset, num_labels, label, save_checkpoint=True):
    print("num labels", num_labels)
    if label in static_trainers:
        return static_trainers[label]

    output_dir = f"./checkpoints/author_classifier_{label}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="F1 Score",
        save_total_limit=1,
        report_to="none",
        logging_steps=10,
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=num_labels
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    if save_checkpoint:
        checkpoint_path = f"./checkpoints/author_classifier_{label}_final"
        trainer.save_model(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"Author classifier saved to {checkpoint_path}")

    static_trainers[label] = trainer
    return trainer


def load_pretrained_author_classifier(label, checkpoint_path=None):
    """
    Load a pretrained author classifier from checkpoint.

    Args:
        label: The label field (e.g., 'author_id')
        checkpoint_path: Path to the checkpoint. If None, uses default path.

    Returns:
        (model, tokenizer) tuple, or (None, None) if checkpoint not found.
    """
    if checkpoint_path is None:
        checkpoint_path = f"./checkpoints/author_classifier_{label}_final"

    try:
        print(f"Loading author classifier from {checkpoint_path}")
        model = BertForSequenceClassification.from_pretrained(checkpoint_path)
        loaded_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        class MockTrainer:
            def __init__(self, model):
                self.model = model

        trainer = MockTrainer(model)
        static_trainers[label] = trainer

        print(f"Successfully loaded author classifier for {label}")
        return model, loaded_tokenizer

    except Exception as e:
        print(f"Error loading author classifier: {e}")
        print(f"Make sure to run pretrain_author_classifier.py first")
        return None, None


def pretrain_author_classifier(
    raw_data_path: str,
    text_field: str = "review",
    label_field: str = "author_id",
    force_retrain: bool = False,
):
    """
    Pretrain the author classification model and save checkpoints.
    Must be run once per dataset before RL training.

    Args:
        raw_data_path:  Path to the original (unmodified) dataset CSV.
        text_field:     Column containing the review/post text.
        label_field:    Column containing integer author IDs.
        force_retrain:  If True, retrain even if checkpoint exists.
    """
    import os

    checkpoint_path = f"./checkpoints/author_classifier_{label_field}_final"
    if os.path.exists(checkpoint_path) and not force_retrain:
        print(f"Checkpoint already exists at {checkpoint_path}. Loading...")
        return load_pretrained_author_classifier(label_field, checkpoint_path)

    print("Starting author classifier pretraining...")
    os.makedirs("./checkpoints", exist_ok=True)

    clean_df = pd.read_csv(raw_data_path)
    total = np.arange(len(clean_df))
    train_idxs, test_idxs = train_test_split(total, test_size=0.10, random_state=42)
    train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.10, random_state=42)

    train_df = clean_df[clean_df.index.isin(train_idxs)][[text_field, label_field]].dropna()
    val_df   = clean_df[clean_df.index.isin(val_idxs)][[text_field, label_field]].dropna()

    tokenized_train = tokenizer(
        train_df[text_field].tolist(),
        return_tensors="pt", padding=True, truncation=True, max_length=max_length,
    )
    tokenized_val = tokenizer(
        val_df[text_field].tolist(),
        return_tensors="pt", padding=True, truncation=True, max_length=max_length,
    )

    num_classes  = train_df[label_field].nunique()
    train_labels = np.array(train_df[label_field])
    val_labels   = np.array(val_df[label_field])

    train_dataset = Dataset.from_dict({
        "input_ids":      tokenized_train["input_ids"],
        "attention_mask": tokenized_train["attention_mask"],
        "labels":         train_labels,
    })
    val_dataset = Dataset.from_dict({
        "input_ids":      tokenized_val["input_ids"],
        "attention_mask": tokenized_val["attention_mask"],
        "labels":         val_labels,
    })

    trainer = get_trained_trainer(
        train_dataset, val_dataset,
        num_labels=num_classes, label=label_field, save_checkpoint=True,
    )
    print("Author classifier pretraining completed!")
    return trainer.model, tokenizer
