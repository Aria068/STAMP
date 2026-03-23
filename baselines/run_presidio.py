"""
run_presidio.py — Presidio NER-based PII anonymisation baseline.

Detects named entities (PERSON, ORG, LOCATION, EMAIL, PHONE_NUMBER, etc.)
using Microsoft Presidio and replaces them with <ENTITY_TYPE> placeholders.
This is a rule-based, deterministic baseline that provides hard entity removal
but no style transformation or utility preservation.

Install:
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_lg

Usage:
    cd Others/Presidio

    # Yelp
    python run_presidio.py \\
        --input  ../../clean_data/yelp_test_split.csv \\
        --output_dir ../../results \\
        --text_field review

    # SynthPAI
    python run_presidio.py \\
        --input  ../../clean_data/SynthPAI_AuthorInfo_test.csv \\
        --output_dir ../../results \\
        --text_field text

Output files (written to --output_dir):
    presidio_<dataset_stem>.csv    — one column: text_field with anonymised texts
    presidio_<dataset_stem>.p      — pickle {"origin": [...], "synth": [...]}
"""

import argparse
import os
import pickle as pkl

import pandas as pd
from tqdm import tqdm

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


def load_engines():
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def anonymize_text(text: str, analyzer, anonymizer, language: str = "en") -> str:
    """Replace all detected PII spans with <ENTITY_TYPE> tokens."""
    results = analyzer.analyze(text=text, language=language)
    if not results:
        return text
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def main():
    parser = argparse.ArgumentParser(
        description="Presidio NER anonymisation baseline."
    )
    parser.add_argument("--input", required=True,
                        help="Path to input CSV.")
    parser.add_argument("--output_dir", default="../../results",
                        help="Directory for output files (default: ../../results).")
    parser.add_argument("--text_field", default="review",
                        help="Column name for text (default: review).")
    parser.add_argument("--language", default="en",
                        help="Language code for Presidio (default: en).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_stem = os.path.splitext(os.path.basename(args.input))[0]

    df = pd.read_csv(args.input)
    if args.text_field not in df.columns:
        raise ValueError(
            f"Column '{args.text_field}' not found in {args.input}. "
            f"Available: {list(df.columns)}"
        )

    texts = df[args.text_field].fillna("").tolist()
    print(f"Input:      {args.input}  ({len(texts)} rows)")
    print(f"Text field: {args.text_field}  |  Language: {args.language}")

    analyzer, anonymizer = load_engines()

    original_texts, synthetic_texts = [], []
    for text in tqdm(texts, desc="Presidio anonymisation"):
        original_texts.append(text)
        synthetic_texts.append(anonymize_text(text, analyzer, anonymizer, args.language))

    csv_out = os.path.join(args.output_dir, f"presidio_{output_stem}.csv")
    pkl_out = os.path.join(args.output_dir, f"presidio_{output_stem}.p")

    pd.DataFrame({args.text_field: synthetic_texts}).to_csv(csv_out, index=False)
    with open(pkl_out, "wb") as f:
        pkl.dump({"origin": original_texts, "synth": synthetic_texts}, f)

    print(f"Saved CSV    → {csv_out}")
    print(f"Saved pickle → {pkl_out}")


if __name__ == "__main__":
    main()
