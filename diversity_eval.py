import sacrebleu
import re
from typing import List
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def self_bleu(synthetic_texts, ngram_order: int = 4) -> float:
    """
    Returns the self-BLEU score (0-100 range), with lower values indicating higher diversity.
    """
    if len(synthetic_texts) <= 1:
        return 0.0

    cfg = dict(
        smooth_method="exp",
        use_effective_order=True,
        lowercase=True,
        tokenize="13a",
    )

    scores = []
    print("Processing Self-Bleu 4-gram")
    for i, hyp in tqdm(enumerate(synthetic_texts), total=len(synthetic_texts)):
        refs = synthetic_texts[:i] + synthetic_texts[i+1:]  # Create references by excluding the hypothesis
        score = sacrebleu.sentence_bleu(hyp, refs).score
        # weights = (1.0, 0, 0, 0, 0.0)  # only 4-gram
        # score = sentence_bleu(refs, hyp, smoothing_function=SmoothingFunction().method1)
        # bleu = sacrebleu.sentence_bleu(hyp, refs, **cfg).score  # Pass the hypothesis as a list
        scores.append(score)

    avg_bleu = sum(scores) / len(scores)  # Calculate the average BLEU score
    return 1 - avg_bleu/100


def compute_distinct_n(
    texts: List[str],
    *,
    n: int = 2,
    lowercase: bool = True,
    strip_punct: bool = True,
) -> float:
    """
    Distinct-n = (# unique n-grams) / (# total n-grams) on the whole corpus.

    • n = 1 ⇒ Distinct-1 (unigrams)
    • n = 2 ⇒ Distinct-2 (bigrams)
    • Return value in [0,1]; higher ⇒ more lexical diversity.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    def preprocess(t: str) -> List[str]:
        if lowercase:
            t = t.lower()
        if strip_punct:
            t = re.sub(r"[^\w\s]", "", t)  # Remove punctuation
        return t.split()

    total_ngrams = 0
    uniq_ngrams = set()

    for line in texts:
        tokens = preprocess(line)
        if len(tokens) < n:
            continue
        total_ngrams += len(tokens) - n + 1  # Count total n-grams
        for i in range(len(tokens) - n + 1):
            uniq_ngrams.add(
                tuple(tokens[i : i + n])
            )  # Add n-grams as tuples to the set

    return len(uniq_ngrams) / total_ngrams if total_ngrams else 0.0


if __name__ == "__main__":
    import sys
    import pickle as pkl

    # Load the data from a pickle file
    data = pkl.load(open(sys.argv[1], "rb"))

    synthetic_texts = []
    for i in range(len(data["synth"])):
        if len(data["synth"][i]) > 0:
            synthetic_texts.append(data["synth"][i])

    # synthetic_texts, ref = [], []

    # data = pkl.load(open("./final_train_data.p", "rb"))
    # for d in data:
    #     synthetic_texts.append(d[-1])
    #     ref.append(d[0])

    print("Eval dataset size is ", len(synthetic_texts))
    # Compute Distinct-n score
    distinct_2_score = compute_distinct_n(
        synthetic_texts, n=2
    )  # Change n for different n-grams
    print(f"Distinct-2 score: {distinct_2_score:.4f}")

    distinct_4_score = compute_distinct_n(
        synthetic_texts, n=4
    )  # Change n for different n-grams
    print(f"Distinct-4 score: {distinct_4_score:.4f}")

    print((distinct_4_score + distinct_2_score)/ 2)

    # Compute self-BLEU score
    self_bleu_score = self_bleu(synthetic_texts[:100])
    print(f"Self_bleu_score : {self_bleu_score}")
