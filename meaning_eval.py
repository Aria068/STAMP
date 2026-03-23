#!/usr/bin/env python
"""
evaluate_text_quality.py
-----------------------------------------------------------------
• compute_bertscore():  returns mean F1 BERTScore
• compute_mauve():      returns MAUVE score
-----------------------------------------------------------------
pickle file must contain:
    - "origin" : List[str]  (reference texts)
    - "synth"  : List[str]  (synthetic texts)
-----------------------------------------------------------------
"""

import sys, pickle as pkl
from pathlib import Path
import torch
from bert_score import BERTScorer
import mauve  # pip install mauve-text


# ────────────────────────────── Function 1 ── BERTScore ─────────────────────
def compute_bertscore(
    synth_texts,
    ref_texts,
    *,
    model_type: str = "roberta-large",
    lang: str = "en",
    rescale_with_baseline: bool = True,
) -> float:
    """Return mean F1 BERTScore between synth_texts and ref_texts."""
    scorer = BERTScorer(
        model_type=model_type,
        lang=lang,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # rescale_with_baseline=rescale_with_baseline,
    )
    _, _, f1 = scorer.score(synth_texts, ref_texts)
    return f1.mean().item()


# ─────────────────────────────── Function 2 ── MAUVE ────────────────────────
def compute_mauve_score(
    synth_texts,
    ref_texts,
    *,
    device_id: int | None = None,
) -> float:
    """Return MAUVE score between synth_texts and ref_texts."""
    if device_id is None:
        device_id = 0 if torch.cuda.is_available() else -1

    result = mauve.compute_mauve(
        p_text=synth_texts,
        q_text=ref_texts,
        device_id=device_id,
        verbose=False,
    )
    return result.mauve


# ─────────────────────────────────── CLI ────────────────────────────────────
def main(pkl_path: str | Path = None):
    data = pkl.load(open(pkl_path, "rb"))
    original_texts = data["origin"]
    synthetic_texts = data["synth"]

    synthetic_texts, original_texts = [], []
    for i in range(len(data["synth"])):
        if len(data["synth"][i]) > 0:
            synthetic_texts.append(data["synth"][i])
            original_texts.append(data["origin"][i])

    # synthetic_texts, original_texts = [], []

    # data = pkl.load(open("./final_train_data.p", "rb"))
    # for d in data:
    #     synthetic_texts.append(d[-1])
    #     original_texts.append(d[0])

    print("Eval dataset size is ", len(original_texts))

    bert = compute_bertscore(synthetic_texts[:500], original_texts[:500])
    mv = compute_mauve_score(synthetic_texts[:500], original_texts[:500])

    print(f"BERTScore (F1, mean) : {bert:.4f}")
    print(f"MAUVE                : {mv:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_text_quality.py <data.pkl>")
        sys.exit(1)

    main(sys.argv[1])
    # main()
