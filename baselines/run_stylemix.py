"""
run_stylemix.py — StyleRemix baseline: LoRA adapter-based stylistic rewriting.

Applies formality-shift LoRA adapters (from Hallisky/StyleRemix) to each
text in a CSV, saving results in the same CSV+pickle format as generate.py.

Model used: NousResearch/Meta-Llama-3-8B-Instruct with formality LoRA adapters.
Only formality adapters are active (other StyleRemix axes are commented out
because they are not relevant to the privacy-rewriting task).

Usage:
    cd Others/StyMix
    python run_stylemix.py \\
        --input   ../../clean_data/yelp_test_split.csv \\
        --output_dir results \\
        --text_field review \\
        --formality 0.8

    python run_stylemix.py \\
        --input   ../../clean_data/imdb_test_split.csv \\
        --output_dir results --text_field review --formality 0.8
"""

import argparse
import os
import pickle as pkl

import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./cache/"

MODEL_PATHS = {
    "formality_more": "hallisky/lora-formality-formal-llama-3-8b",
    "formality_less": "hallisky/lora-formality-informal-llama-3-8b",
}
FIRST_MODEL = "formality_more"
MAX_NEW_TOKENS = 1024
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"


def convert_data_to_format(text: str) -> str:
    return f"### Original: {text}\n ### Rewrite:"


def remix(model, tokenizer, input_text: str, formality: float) -> str:
    """Apply formality LoRA adapter and generate a rewrite."""
    device = model.device

    sliders_dict = {}
    if formality > 0:
        sliders_dict["formality_more"] = abs(formality)
    elif formality < 0:
        sliders_dict["formality_less"] = abs(formality)

    if not sliders_dict:
        return input_text

    combo_name = "".join(
        f"{k}{int(100 * v)}-" for k, v in sliders_dict.items()
    ).rstrip("-")

    model.add_weighted_adapter(
        list(sliders_dict.keys()),
        weights=list(sliders_dict.values()),
        adapter_name=combo_name,
        combination_type="cat",
    )
    model.set_adapter(combo_name)

    converted = convert_data_to_format(input_text)
    inputs = tokenizer(
        converted, return_tensors="pt", max_length=2048, truncation=True
    ).to(device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, top_p=0.95)

    return tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()


def load_model(device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, add_bos_token=True, add_eos_token=False, padding_side="left"
    )
    tokenizer.add_special_tokens({"pad_token": "<padding_token>"})

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(
        base_model, MODEL_PATHS[FIRST_MODEL], adapter_name=FIRST_MODEL
    )
    for name, path in MODEL_PATHS.items():
        if name != FIRST_MODEL:
            model.load_adapter(path, adapter_name=name)

    model.to(device)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="StyleRemix baseline: formality-shift LoRA rewriting."
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
        "--formality", type=float, default=0.8,
        help=(
            "Formality slider: positive → more formal, negative → less formal. "
            "Range [-1, 1]. Default: 0.8."
        ),
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Limit processing to the first N rows (optional).",
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
    if args.sample is not None:
        df = df.iloc[: args.sample]

    print(f"Input:      {args.input}  ({len(df)} rows)")
    print(f"Text field: {args.text_field}")
    print(f"Formality:  {args.formality}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model, tokenizer = load_model(device)

    original_texts, synthetic_texts = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="StyleRemix rewriting"):
        text = str(row[args.text_field])
        rewrite = remix(model, tokenizer, text, formality=args.formality)
        original_texts.append(text)
        synthetic_texts.append(rewrite)

    formality_tag = f"f{int(args.formality * 100):+d}".replace("+", "p").replace("-", "m")
    csv_out = os.path.join(
        args.output_dir, f"stylemix_{output_stem}_{formality_tag}.csv"
    )
    pkl_out = os.path.join(
        args.output_dir, f"stylemix_{output_stem}_{formality_tag}.p"
    )

    pd.DataFrame({args.text_field: synthetic_texts}).to_csv(csv_out, index=False)
    with open(pkl_out, "wb") as f:
        pkl.dump({"origin": original_texts, "synth": synthetic_texts}, f)

    print(f"Saved CSV    → {csv_out}")
    print(f"Saved pickle → {pkl_out}")


if __name__ == "__main__":
    main()
