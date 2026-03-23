"""
sft_data_gen.py — Stage 1a: Teacher-guided pseudo-parallel SFT corpus generation.

For each source sentence:
  1. Detect whether it is a stylistic outlier via GlobalStyleMemory.
  2. Sample an adaptive style reference (common if outlier, diverse if not).
  3. Feed source + reference to a frozen large teacher LLM with randomised prompts.
  4. Generate N candidate rewrites; keep those with BERTScore(candidate, source) >= threshold.
  5. Among passing candidates, select the one whose style embedding is closest to the reference.
  6. Save (source, best_rewrite, ref_text, is_outlier) as a pseudo-parallel CSV.

Run before sft_train.py:
    python sft_data_gen.py --dataset yelp --output clean_data/yelp_sft_pairs.csv --sample 5000

Paper reference: §3.2 "GlobalMem-Guided Unsupervised Data Generation and LLM Fine-Tuning"
"""

import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from bert_score import BERTScorer
from sklearn.metrics.pairwise import cosine_distances
from vllm import LLM, SamplingParams

from utils import GlobalStyleMemory, _STYLE_EMBEDDER
from dataset import DATASET_PATH
from config import GSM_LAMBDA, SEM_THRESHOLD, TEACHER_MODEL


# ── Randomised teacher prompts (paper §3.2) ───────────────────────────────────
# Three distinct phrasings to encourage diverse generation.
TEACHER_PROMPTS = [
    (
        "Rewrite the following text to protect the author's privacy. "
        "Remove all personal identifiers. Adopt a style similar to the reference.\n"
        "Source: {src}\nStyle reference: {ref}\nRewrite:"
    ),
    (
        "Paraphrase the source text in the tone of the reference excerpt. "
        "Replace any identifying information with plausible alternatives.\n"
        "Source: {src}\nReference style: {ref}\nParaphrased output:"
    ),
    (
        "Transform the source to match the writing style of the reference "
        "while suppressing all personal details.\n"
        "Source: {src}\nReference: {ref}\nTransformed text:"
    ),
]


def build_sft_corpus(
    sentences: list,
    memory: GlobalStyleMemory,
    teacher_llm,
    bert_scorer: BERTScorer,
    sem_threshold: float = 0.85,
    num_candidates: int = 4,
) -> list:
    """
    Build pseudo-parallel (source, best_rewrite) pairs.

    Key design decisions vs v1:
    - Style distance is measured from EACH CANDIDATE to the REFERENCE embedding,
      not to the global pool mean. Paper §3.2: "minimizing style distance."
    - Reference is sampled adaptively: common node for outliers, diverse node
      for in-distribution texts.

    Returns:
        List of dicts with keys: review, answer, ref_text, is_outlier.
    """
    pairs = []
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

    for src in tqdm(sentences, desc="Generating SFT pairs"):
        is_out, _ = memory.is_outlier(src)

        # Adaptively sample style reference
        ref_text = (
            memory.sample_reference_common()
            if is_out
            else memory.sample_reference_diverse(src)
        )

        # Embed reference once for later distance computation
        ref_emb = np.asarray(
            _STYLE_EMBEDDER.encode(
                [ref_text], convert_to_numpy=True, show_progress_bar=False
            )
        )  # shape (1, dim)

        # Generate candidates using randomised prompts
        prompts = [
            random.choice(TEACHER_PROMPTS).format(src=src[:300], ref=ref_text[:120])
            for _ in range(num_candidates)
        ]
        outputs = teacher_llm.generate(prompts, sampling_params)
        candidates = [o.outputs[0].text.strip() for o in outputs]

        # Filter by semantic fidelity to the source
        _, _, f1s = bert_scorer.score(candidates, [src] * len(candidates))
        valid_candidates = [
            c for c, f in zip(candidates, f1s) if float(f) >= sem_threshold
        ]
        if not valid_candidates:
            continue   # discard example if no candidate passes the threshold

        # Among valid, pick the one closest in style to the reference embedding
        cand_embs = np.asarray(
            _STYLE_EMBEDDER.encode(
                valid_candidates, convert_to_numpy=True, show_progress_bar=False
            )
        )  # shape (n_valid, dim)
        style_dists_to_ref = cosine_distances(cand_embs, ref_emb).flatten()
        best_idx = int(np.argmin(style_dists_to_ref))
        best_candidate = valid_candidates[best_idx]

        pairs.append({
            "review": src,
            "answer": best_candidate,
            "ref_text": ref_text,
            "is_outlier": int(is_out),
        })

    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1a: Generate pseudo-parallel SFT corpus via teacher LLM."
    )
    parser.add_argument("--dataset", default="yelp",
                        help="Dataset key from DATASET_PATH (yelp/tweet/imdb/synpai).")
    parser.add_argument("--teacher_model", default=TEACHER_MODEL,
                        help="Path or HuggingFace ID of the teacher model.")
    parser.add_argument("--output", default="clean_data/yelp_sft_pairs.csv",
                        help="Output CSV path for pseudo-parallel pairs.")
    parser.add_argument("--sample", type=int, default=5000,
                        help="Number of source sentences to sample.")
    parser.add_argument("--tensor_parallel", type=int, default=2,
                        help="Tensor parallelism for vLLM (number of GPUs).")
    parser.add_argument("--num_candidates", type=int, default=4,
                        help="Teacher rewrites per source sentence.")
    args = parser.parse_args()

    if args.dataset not in DATASET_PATH:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choose from: {list(DATASET_PATH.keys())}")

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(DATASET_PATH[args.dataset])
    sentences = df["review"].dropna().tolist()
    random.shuffle(sentences)
    sentences = sentences[: args.sample]
    print(f"Using {len(sentences)} source sentences.")

    # Build GlobalStyleMemory from source corpus
    print("Building GlobalStyleMemory...")
    memory = GlobalStyleMemory(lambda_sensitivity=GSM_LAMBDA)
    memory.fit(sentences)
    print(f"Memory built. τ = {memory.tau:.4f}")

    # Load teacher LLM (requires multi-GPU for 70B model)
    print(f"Loading teacher model: {args.teacher_model}")
    teacher = LLM(
        model=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel,
    )

    bert_scorer = BERTScorer(model_type="roberta-large", lang="en")

    print("Generating SFT pairs...")
    pairs = build_sft_corpus(
        sentences=sentences,
        memory=memory,
        teacher_llm=teacher,
        bert_scorer=bert_scorer,
        sem_threshold=SEM_THRESHOLD,
        num_candidates=args.num_candidates,
    )

    out_df = pd.DataFrame(pairs)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(pairs)} SFT pairs to {args.output} "
          f"({len(pairs)/len(sentences)*100:.1f}% yield)")
