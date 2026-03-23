"""
run_dpmlm.py — DP-MLM baseline: privacy-preserving text rewriting via
differential privacy masked language modelling.

Outputs a CSV and pickle in the same format as generate.py so all
eval scripts can consume it without modification.

Usage:
    cd Others/DPMLM
    python run_dpmlm.py \\
        --input  ../../clean_data/yelp_test_split.csv \\
        --output_dir results \\
        --text_field review \\
        --epsilon 50

    # IMDb
    python run_dpmlm.py \\
        --input  ../../clean_data/imdb_test_split.csv \\
        --output_dir results --text_field review --epsilon 50

    # Twitter
    python run_dpmlm.py \\
        --input  ../../clean_data/tweet_review.csv \\
        --output_dir results --text_field review --epsilon 50
"""

import argparse
import os
import pickle as pkl

import pandas as pd
from tqdm import tqdm

import DPMLM


def main():
    parser = argparse.ArgumentParser(
        description="DP-MLM baseline: differentially private text rewriting."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output_dir", default="results",
        help="Directory to save output CSV and pickle (default: results).",
    )
    parser.add_argument(
        "--text_field", default="review",
        help="Column name for text to rewrite (default: review).",
    )
    parser.add_argument(
        "--epsilon", type=float, default=50,
        help="Privacy budget ε for DP-MLM (default: 50).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_stem = os.path.splitext(os.path.basename(args.input))[0]

    df = pd.read_csv(args.input)
    if args.text_field not in df.columns:
        raise ValueError(
            f"Column '{args.text_field}' not found in {args.input}. "
            f"Available: {list(df.columns)}"
        )

    print(f"Input:      {args.input}  ({len(df)} rows)")
    print(f"Text field: {args.text_field}")
    print(f"Epsilon:    {args.epsilon}")

    M = DPMLM.DPMLM()

    # RoBERTa max is 512 tokens. With CONCAT=True, privatize() encodes
    # original + masked sentence together (~2× length + 3 special tokens).
    # We truncate to MAX_SUBWORD_TOKENS subword tokens per text so that
    # 2 × MAX_SUBWORD_TOKENS + 3 ≤ 512  →  MAX_SUBWORD_TOKENS = 240.
    MAX_SUBWORD_TOKENS = 240

    def truncate_to_token_limit(text, tokenizer, max_tokens):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    original_texts, synthetic_texts = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="DP-MLM rewriting"):
        original = row[args.text_field]
        text = truncate_to_token_limit(original, M.tokenizer, MAX_SUBWORD_TOKENS)
        try:
            res = M.dpmlm_rewrite(text, epsilon=args.epsilon)
            synthetic_texts.append(res[0])
        except Exception as e:
            print(f"\nWarning: skipping row, dpmlm_rewrite failed: {e}")
            synthetic_texts.append(text)   # fall back to truncated original
        original_texts.append(original)

    eps_tag = int(args.epsilon)
    csv_out = os.path.join(
        args.output_dir, f"dpmlm_{output_stem}_e{eps_tag}.csv"
    )
    pkl_out = os.path.join(
        args.output_dir, f"dpmlm_{output_stem}_e{eps_tag}.p"
    )

    pd.DataFrame({args.text_field: synthetic_texts}).to_csv(csv_out, index=False)
    with open(pkl_out, "wb") as f:
        pkl.dump({"origin": original_texts, "synth": synthetic_texts}, f)

    print(f"Saved CSV    → {csv_out}")
    print(f"Saved pickle → {pkl_out}")


if __name__ == "__main__":
    main()
