"""
run_tarot.py — TAROT-DPO baseline: authorship obfuscation via GPT-2 DPO.

Model: gabrielloiseau/TAROT-DPO
Paper: https://arxiv.org/abs/2407.21630

Fine-tuned GPT-2 (0.4B) for privacy-preserving text rewriting using Direct
Preference Optimization with LUAR-MUD and gte-large-en-v1.5 reward models.

Outputs a CSV and pickle in the same format as generate.py / run_dpmlm.py so
all eval scripts (privacy_eval.py, diversity_eval.py, meaning_eval.py) and
eval_all.sh can consume the results without modification.

Usage:
    cd Others/TAROT

    # Yelp
    python run_tarot.py \\
        --input  ../../clean_data/yelp_test_split.csv \\
        --output_dir ../../results \\
        --text_field review

    # Twitter
    python run_tarot.py \\
        --input  ../../clean_data/twitter_test.csv \\
        --output_dir ../../results \\
        --text_field review

    # IMDb
    python run_tarot.py \\
        --input  ../../clean_data/imdb_test_split.csv \\
        --output_dir ../../results \\
        --text_field review \\
        --batch_size 4 --max_new_tokens 256

Output files (written to --output_dir):
    tarot_<dataset_stem>.csv    — one column: text_field with synthetic texts
    tarot_<dataset_stem>.p      — pickle {"origin": [...], "synth": [...]}
"""

import argparse
import os
import pickle as pkl

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "gabrielloiseau/TAROT-DPO"
# Sentinel appended to each input, as specified by the model card
_EOT = "<|endoftext|>"


def load_model(device: str):
    print(f"Loading {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # GPT-2 has no pad token by default; use eos so batched generation works.
    # Left-padding is required for decoder-only batched generation.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}.")
    return tokenizer, model


def rewrite_batch(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
) -> list[str]:
    """Rewrite a batch of texts with TAROT-DPO."""
    prompts = [t + _EOT for t in texts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Slice off the prompt tokens; decode only the generated continuation
    gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def main():
    parser = argparse.ArgumentParser(
        description="TAROT-DPO baseline: authorship obfuscation via GPT-2 DPO."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output_dir", default="../../results",
        help="Directory to save output CSV and pickle (default: ../../results).",
    )
    parser.add_argument(
        "--text_field", default="review",
        help="Column name for text to rewrite (default: review).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for generation (default: 8).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="Maximum new tokens to generate per sample (default: 128).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: 'cuda', 'cpu', etc. Auto-detected if not set.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"Text field: {args.text_field}")
    print(f"Batch size: {args.batch_size}  |  max_new_tokens: {args.max_new_tokens}")

    tokenizer, model = load_model(device)

    original_texts, synthetic_texts = [], []
    batches = [texts[i:i + args.batch_size] for i in range(0, len(texts), args.batch_size)]

    for batch in tqdm(batches, desc="TAROT-DPO rewriting"):
        rewrites = rewrite_batch(batch, tokenizer, model, device, args.max_new_tokens)
        original_texts.extend(batch)
        synthetic_texts.extend(rewrites)

    csv_out = os.path.join(args.output_dir, f"tarot_{output_stem}.csv")
    pkl_out = os.path.join(args.output_dir, f"tarot_{output_stem}.p")

    pd.DataFrame({args.text_field: synthetic_texts}).to_csv(csv_out, index=False)
    with open(pkl_out, "wb") as f:
        pkl.dump({"origin": original_texts, "synth": synthetic_texts}, f)

    print(f"Saved CSV    → {csv_out}")
    print(f"Saved pickle → {pkl_out}")


if __name__ == "__main__":
    main()
