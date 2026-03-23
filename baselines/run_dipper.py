"""
run_dipper.py — DIPPER paraphrase baseline.

Model: kalpeshk2011/dipper-paraphraser-xxl  (T5-11B)
Paper: Krishna et al. "Paraphrasing evades detectors of AI-generated text, but
       retrieval is an effective defense", NeurIPS 2023.

DIPPER rewrites text sentence-by-sentence using controllable lexical (lex) and
order (order) diversity codes (integers in {0, 20, 40, 60, 80, 100}).
For the privacy baseline we use lex=60, order=0 — high lexical diversity
with preserved sentence order — following the settings in the privacy
obfuscation literature.

Memory note: T5-11B requires ~22 GB in fp32 / ~11 GB in fp16 / ~6 GB in int8.
Use --load_in_8bit on machines with ≤16 GB VRAM.

Install:
    pip install transformers sentencepiece nltk accelerate bitsandbytes

Usage:
    cd Others/DIPPER

    # Yelp (fp16, batch_size=4)
    python run_dipper.py \\
        --input  ../../clean_data/yelp_test_split.csv \\
        --output_dir ../../results \\
        --text_field review

    # Low-VRAM (int8)
    python run_dipper.py \\
        --input  ../../clean_data/yelp_test_split.csv \\
        --output_dir ../../results \\
        --text_field review \\
        --load_in_8bit

Output files (written to --output_dir):
    dipper_<dataset_stem>.csv    — one column: text_field with paraphrased texts
    dipper_<dataset_stem>.p      — pickle {"origin": [...], "synth": [...]}
"""

import argparse
import os
import pickle as pkl

import torch
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_ID = "kalpeshk2011/dipper-paraphraser-xxl"


def load_model(load_in_8bit: bool, device: str):
    print(f"Loading {MODEL_ID} …")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    if load_in_8bit:
        model = T5ForConditionalGeneration.from_pretrained(
            MODEL_ID, load_in_8bit=True, device_map="auto"
        )
    else:
        dtype = torch.float16 if device != "cpu" else torch.float32
        model = T5ForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=dtype
        ).to(device)
    model.eval()
    print(f"Model loaded.  8bit={load_in_8bit}  device={device}")
    return tokenizer, model


def paraphrase(
    text: str,
    tokenizer,
    model,
    device: str,
    lex_diversity: int = 60,
    order_diversity: int = 0,
    sent_interval: int = 3,
    max_length: int = 512,
) -> str:
    """
    Rewrite `text` in sliding windows of `sent_interval` sentences.

    Input format matches the DIPPER model card:
        "lexical = {lex}, order = {order} <sent> {window} </sent>"

    The previous output is prepended as context on each subsequent window
    (as in the original DIPPER paper), giving the model continuity over long docs.
    """
    sentences = sent_tokenize(text)
    prefix = ""
    output_parts = []

    for i in range(0, len(sentences), sent_interval):
        window = " ".join(sentences[i: i + sent_interval])
        input_text = f"lexical = {lex_diversity}, order = {order_diversity}"
        if prefix:
            input_text += f" {prefix}"
        input_text += f" <sent> {window} </sent>"

        enc = tokenizer(
            [input_text],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                do_sample=True,
                top_p=0.75,
                top_k=None,
                max_new_tokens=max_length,
            )

        out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        prefix = out_text       # carry forward for context
        output_parts.append(out_text)

    return " ".join(output_parts).strip()


def main():
    parser = argparse.ArgumentParser(
        description="DIPPER paraphrase baseline for privacy evaluation."
    )
    parser.add_argument("--input", required=True,
                        help="Path to input CSV.")
    parser.add_argument("--output_dir", default="../../results",
                        help="Directory for output files (default: ../../results).")
    parser.add_argument("--text_field", default="review",
                        help="Column name for text (default: review).")
    parser.add_argument("--lex_diversity", type=int, default=60,
                        help="Lexical diversity code {0,20,40,60,80,100} (default: 60).")
    parser.add_argument("--order_diversity", type=int, default=0,
                        help="Order diversity code {0,20,40,60,80,100} (default: 0).")
    parser.add_argument("--sent_interval", type=int, default=3,
                        help="Sentences per rewriting window (default: 3).")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max tokens for encoder input and decoder output (default: 512).")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit (bitsandbytes) for low-VRAM machines.")
    parser.add_argument("--device", default=None,
                        help="Device override (e.g. 'cuda:0', 'cpu'). Auto-detected if unset.")
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
    print(f"Input:           {args.input}  ({len(texts)} rows)")
    print(f"Text field:      {args.text_field}")
    print(f"lex={args.lex_diversity}  order={args.order_diversity}  "
          f"sent_interval={args.sent_interval}")

    tokenizer, model = load_model(args.load_in_8bit, device)

    original_texts, synthetic_texts = [], []
    for text in tqdm(texts, desc="DIPPER paraphrasing"):
        original_texts.append(text)
        synthetic_texts.append(
            paraphrase(
                text, tokenizer, model, device,
                lex_diversity=args.lex_diversity,
                order_diversity=args.order_diversity,
                sent_interval=args.sent_interval,
                max_length=args.max_length,
            )
        )

    csv_out = os.path.join(args.output_dir, f"dipper_{output_stem}.csv")
    pkl_out = os.path.join(args.output_dir, f"dipper_{output_stem}.p")

    pd.DataFrame({args.text_field: synthetic_texts}).to_csv(csv_out, index=False)
    with open(pkl_out, "wb") as f:
        pkl.dump({"origin": original_texts, "synth": synthetic_texts}, f)

    print(f"Saved CSV    → {csv_out}")
    print(f"Saved pickle → {pkl_out}")


if __name__ == "__main__":
    main()
