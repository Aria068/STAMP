"""
mia_attack.py — Membership Inference Attack (MIA) privacy benchmark.

Trains a shadow-model attack (TF-IDF + LR target, RF attacker) on
the original corpus, then tests whether membership can be inferred from
synthetic rewrites. A synthetic MIA AUC near 0.5 indicates strong privacy.

Usage:
    python mia_attack.py \\
        --orig_path  clean_data/yelp_test_split.csv \\
        --syn_path   results/ours_yelp_syn.csv \\
        --text_field review \\
        --label_field sentiment_id
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def train_target_model(texts, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)
    return model, vectorizer


def build_attack_dataset(model, vectorizer, member_texts, nonmember_texts):
    X_member = vectorizer.transform(member_texts)
    X_nonmember = vectorizer.transform(nonmember_texts)
    member_probs = np.max(model.predict_proba(X_member), axis=1)
    nonmember_probs = np.max(model.predict_proba(X_nonmember), axis=1)
    X_attack = np.concatenate(
        [member_probs.reshape(-1, 1), nonmember_probs.reshape(-1, 1)]
    )
    y_attack = np.concatenate(
        [np.ones(len(member_probs)), np.zeros(len(nonmember_probs))]
    )
    return X_attack, y_attack


def train_attack_model(X_attack, y_attack):
    X_train, X_test, y_train, y_test = train_test_split(
        X_attack, y_attack, test_size=0.3, random_state=42
    )
    attacker = RandomForestClassifier(n_estimators=100)
    attacker.fit(X_train, y_train)
    auc = roc_auc_score(y_test, attacker.predict_proba(X_test)[:, 1])
    return attacker, auc


def run_mia_benchmark(real_texts, real_labels, synthetic_texts):
    """
    Train target model on real data, then train a shadow attacker.
    Evaluate on synthetic rewrites as the 'protected' corpus.

    Returns dict with mia_real (baseline leakage) and mia_syn (after rewrite).
    """
    # Use same labels for synthetic (structure mirrors the real corpus)
    synthetic_labels = real_labels[: len(synthetic_texts)]

    real_train, real_test, y_train, y_test = train_test_split(
        real_texts, real_labels, test_size=0.5, random_state=42
    )
    syn_train, _, y_syn_train, _ = train_test_split(
        synthetic_texts, synthetic_labels, test_size=0.5, random_state=42
    )

    real_model, real_vec = train_target_model(real_train, y_train)
    syn_model, syn_vec = train_target_model(syn_train, y_syn_train)

    X_attack_real, y_attack_real = build_attack_dataset(
        real_model, real_vec, real_train, real_test
    )
    X_attack_syn, y_attack_syn = build_attack_dataset(
        syn_model, syn_vec, real_train, real_test
    )

    _, mia_real = train_attack_model(X_attack_real, y_attack_real)
    _, mia_syn = train_attack_model(X_attack_syn, y_attack_syn)

    print("=== MIA Privacy Benchmark (Text Classification) ===")
    print(f"Real Data Model:      {mia_real:.3f}  (higher = more leakage)")
    print(f"Synthetic Data Model: {mia_syn:.3f}  (closer to 0.5 = better privacy)")
    print(f"Privacy Gain:         {round(1 - (mia_syn / mia_real), 3)}")
    return {"mia_real": mia_real, "mia_syn": mia_syn,
            "privacy_gain": round(1 - (mia_syn / mia_real), 3)}


def main():
    parser = argparse.ArgumentParser(
        description="MIA privacy benchmark: measures membership leakage of synthetic text."
    )
    parser.add_argument("--orig_path", required=True,
                        help="CSV with original texts (member corpus).")
    parser.add_argument("--syn_path", required=True,
                        help="CSV with synthetic (rewritten) texts.")
    parser.add_argument("--text_field", default="review",
                        help="Column name for text (default: review).")
    parser.add_argument("--label_field", default="sentiment_id",
                        help="Column name for label (default: sentiment_id).")
    args = parser.parse_args()

    df_orig = pd.read_csv(args.orig_path)
    df_syn = pd.read_csv(args.syn_path)

    for col, df, path in [
        (args.text_field, df_orig, args.orig_path),
        (args.label_field, df_orig, args.orig_path),
        (args.text_field, df_syn, args.syn_path),
    ]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}. "
                             f"Available: {list(df.columns)}")

    real_texts = df_orig[args.text_field].dropna().tolist()
    real_labels = df_orig[args.label_field].dropna().tolist()
    synthetic_texts = df_syn[args.text_field].dropna().tolist()

    # Align lengths
    n = min(len(real_texts), len(synthetic_texts))
    real_texts, real_labels, synthetic_texts = (
        real_texts[:n], real_labels[:n], synthetic_texts[:n]
    )
    print(f"Loaded {n} aligned pairs from {args.orig_path} + {args.syn_path}")

    run_mia_benchmark(real_texts, real_labels, synthetic_texts)


if __name__ == "__main__":
    main()
