#!/usr/bin/env python
"""
privacy_eval.py — Privacy evaluation toolkit for synthetic text.

Metrics included
----------------
• Entity-matching recall
• Outlier detectors   (distance, IsolationForest, HDBSCAN)
• Privacy-Exposure Index  (PEI*)
• Outlier similarity  (nearest synth → real outlier, cosine < 0.1)
• NNDR (nearest-neighbour distance ratio in embedding space)
• Attribute leakage F1
• MIA AUC
"""

from __future__ import annotations
from typing import List, Dict, Any, Literal, Tuple
from collections import namedtuple
import sys
import pickle as pkl
import pathlib as pl

import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# ── Global singletons (heavy models loaded once) ──────────────────────────────
_NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
_EMBEDDER = SentenceTransformer("AnnaWegmann/Style-Embedding")
_BERTEMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


class StyleOutlierDetector:
    """
    Uses any SentenceTransformer-compatible model as a style encoder.
    IsolationForest flags stylistic outliers.
    """

    def __init__(
        self, model_name: str = "AnnaWegmann/Style-Embedding", sigma: float = 2.0
    ):
        self.fitted = False
        self.sigma = sigma
        self.distance_threshold = None
        self.X = None

    def _embed(self, sentences):
        return np.asarray(
            _EMBEDDER.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        )

    def fit(self, texts):
        self.X = self._embed(texts)
        d = cosine_distances(self.X)
        np.fill_diagonal(d, np.nan)
        mean_d = np.nanmean(d, axis=1)
        mu, std = mean_d.mean(), mean_d.std() + 1e-9
        self.distance_threshold = mu + self.sigma * std
        self.fitted = True

    def score(self, text) -> float:
        if not self.fitted:
            raise RuntimeError("Call `fit` with training data first.")
        v = self._embed([text])
        distances = cosine_distances(v, self.X).flatten()
        return np.nanmean(distances), distances

    def is_outlier(self, text) -> bool:
        mean_distance, distances = self.score(text)
        return mean_distance > self.distance_threshold, mean_distance


# ── Utility helpers ───────────────────────────────────────────────────────────

def embed_texts(texts: List[str], *, batch_size: int = 32) -> np.ndarray:
    return _EMBEDDER.encode(
        texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
    )


def semantic_embed_texts(sentences) -> np.ndarray:
    return np.asarray(
        _BERTEMBEDDER.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    )


def cosine_knn(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(metric="cosine", n_neighbors=k).fit(a)
    dists, _ = nn.kneighbors(b)
    return dists


def extract_entities(text: str) -> set[str]:
    return {ent.text for ent in _NLP(text).ents}


# ── Entity-matching recall ────────────────────────────────────────────────────

def entity_recall(synth: List[str], ref: List[str]) -> float:
    """Fraction of reference entities reproduced in synth (lower ⇒ better privacy)."""
    scores, total = 0.0, 0
    for s, r in zip(synth, ref):
        ents_r = extract_entities(r)
        if not ents_r:
            continue
        overlap = len(ents_r & extract_entities(s)) / len(ents_r)
        scores += overlap
        total += 1
    return scores / total if total else 0.0


# ── Outlier detection ─────────────────────────────────────────────────────────

def _distance_density(
    embeds: np.ndarray, *, z_thr: float, radius: float, min_k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    d = cosine_distances(embeds)
    np.fill_diagonal(d, np.nan)
    mean_d = np.nanmean(d, axis=1)
    mean_d_global = mean_d.mean()
    mean_d_std = mean_d.std()
    z_scores = (mean_d - mean_d_global) / (mean_d_std + 1e-9)
    is_outlier = z_scores >= z_thr
    neighbors = np.nansum(d < radius, axis=1).astype(int)
    privacy_risk = np.logical_and(is_outlier, neighbors < min_k)
    min_d = np.nanmin(d, axis=1)
    outlier_dist_mean = np.mean(min_d[is_outlier])
    return is_outlier, neighbors, privacy_risk, outlier_dist_mean


def _isoforest_density(
    embeds: np.ndarray, *, radius: float, min_k: int, **kw
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iso = IsolationForest(**kw).fit(embeds)
    out = iso.predict(embeds) == -1
    d = cosine_distances(embeds)
    np.fill_diagonal(d, np.inf)
    neigh = (d < radius).sum(1)
    risk = np.logical_and(out, neigh < min_k)
    return out, neigh, risk


def _hdbscan_density(
    embeds: np.ndarray,
    *,
    radius: float,
    min_k: int,
    min_cluster_size: int = 5,
    outlier_p: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clusterer = HDBSCAN(metric="cosine", min_cluster_size=min_cluster_size).fit(embeds)
    noise_prob = 1.0 - clusterer.probabilities_
    out = np.logical_or(clusterer.labels_ == -1, noise_prob > outlier_p)
    d = cosine_distances(embeds)
    np.fill_diagonal(d, np.inf)
    neigh = (d < radius).sum(1)
    risk = np.logical_and(out, neigh < min_k)
    return out, neigh, risk


def detect_outliers(
    texts: List[str],
    method: Literal["distance", "isoforest", "hdbscan"] = "distance",
    *,
    radius: float = 0.25,
    min_neighbors: int = 5,
    z_thr: float = 2.0,
    iforest_kw: Dict[str, Any] | None = None,
    hdb_kw: Dict[str, Any] | None = None,
    return_embeds: bool = False,
) -> Dict[str, Any]:
    embeds = embed_texts(texts)
    if method == "distance":
        out, neigh, risk, outlier_dist_mean = _distance_density(
            embeds, z_thr=z_thr, radius=radius, min_k=min_neighbors
        )
    elif method == "isoforest":
        out, neigh, risk = _isoforest_density(
            embeds, radius=radius, min_k=min_neighbors,
            **(iforest_kw or {"random_state": 0}),
        )
        outlier_dist_mean = float("nan")
    elif method == "hdbscan":
        out, neigh, risk = _hdbscan_density(
            embeds, radius=radius, min_k=min_neighbors, **(hdb_kw or {})
        )
        outlier_dist_mean = float("nan")
    else:
        raise ValueError("method must be 'distance', 'isoforest', or 'hdbscan'")

    res = {
        "is_outlier": out,
        "neighbors": neigh,
        "privacy_risk": risk,
        "outlier_dist_mean": outlier_dist_mean,
    }
    if return_embeds:
        res["embeddings"] = embeds
    return res


# ── Privacy-Exposure Index (PEI*) ─────────────────────────────────────────────

def privacy_exposure_index(
    is_outlier: np.ndarray, neighbors: np.ndarray, *, k_thr: int = 5
) -> float:
    iso = is_outlier.astype(float)
    isolation = 1.0 - np.clip(neighbors / k_thr, 0, 1)
    return float((iso * isolation).sum() / len(is_outlier))


# ── Outlier similarity ────────────────────────────────────────────────────────

def outlier_similarity(real: List[str], synth: List[str]) -> float:
    det_r = StyleOutlierDetector()
    det_r.fit(real)
    r_vecs, r_text = [], []
    for t in real:
        is_out, _ = det_r.is_outlier(t)
        if is_out:
            r_text.append(t)
            r_vecs.append(semantic_embed_texts(t))

    det_t = StyleOutlierDetector()
    det_t.fit(synth)
    s_vecs, s_text = [], []
    for t in synth:
        is_out, _ = det_t.is_outlier(t)
        if is_out:
            s_vecs.append(semantic_embed_texts(t))
            s_text.append(t)

    if not r_vecs or not s_vecs:
        return 0.0

    dist = cosine_knn(np.vstack(r_vecs), np.vstack(s_vecs), k=1)
    exact = (dist[:, 0] < 0.1).sum()
    return exact / len(r_vecs)


# ── NNDR in embedding space ───────────────────────────────────────────────────

def nndr_embedding(real: List[str], synth: List[str]) -> namedtuple:
    real_v, synth_v = embed_texts(real), embed_texts(synth)
    dist = cosine_knn(real_v, synth_v, k=2)
    ratios = dist[:, 0] / (dist[:, 1] + 1e-12)
    NNDR = namedtuple("NNDRResult", ("min", "mean", "std"))
    return NNDR(ratios.min(), ratios.mean(), ratios.std())


# ── PCA plot helper ───────────────────────────────────────────────────────────

def plot_pca(embeds: np.ndarray, outlier_mask: np.ndarray, path):
    pca = PCA(n_components=2).fit_transform(embeds)
    plt.scatter(*pca[~outlier_mask].T, s=8, alpha=0.6, label="inliers")
    plt.scatter(*pca[outlier_mask].T, s=18, marker="x", label="outliers")
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ── Attribute leakage F1 ──────────────────────────────────────────────────────

def attribute_leakage_f1(
    synth_texts: List[str],
    labels: List[int],
    classifier,
    tokenizer_,
    max_len: int = 512,
) -> float:
    """
    Macro-F1 of a frozen BERT attribute classifier on rewritten text.
    Lower = better privacy (harder to infer author / gender / age).
    Paper §4.4: Author-ID F1, Gender F1, Age F1.
    """
    import torch
    from sklearn.metrics import f1_score
    encodings = tokenizer_(
        synth_texts, truncation=True, padding=True,
        max_length=max_len, return_tensors="pt",
    )
    with torch.no_grad():
        logits = classifier(**encodings).logits
    preds = logits.argmax(dim=-1).cpu().numpy()
    return float(f1_score(labels, preds, average="macro"))


# ── MIA AUC ───────────────────────────────────────────────────────────────────

def mia_auc(
    member_texts: List[str],
    nonmember_texts: List[str],
    shadow_model,
    tokenizer_,
) -> float:
    """
    Likelihood-ratio membership inference attack AUC.
    Lower AUC = better privacy (attack is no better than chance).
    Paper §4.4: MIA_AUC metric.
    """
    import torch
    from sklearn.metrics import roc_auc_score
    scores, labels = [], []
    for text, label in (
        [(t, 1) for t in member_texts] + [(t, 0) for t in nonmember_texts]
    ):
        enc = tokenizer_(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            loss = shadow_model(**enc, labels=enc["input_ids"]).loss
        scores.append(-loss.item())
        labels.append(label)
    return float(roc_auc_score(labels, scores))


# ── Self-BLEU diversity ───────────────────────────────────────────────────────

def self_bleu_score(texts: List[str], sample_n: int = 200) -> float:
    """Corpus-level self-BLEU. Lower = more lexically diverse outputs."""
    import random
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    if len(texts) > sample_n:
        texts = random.sample(texts, sample_n)
    tokenised = [t.split() for t in texts]
    scores = []
    for i, hyp in enumerate(tokenised):
        refs = [t for j, t in enumerate(tokenised) if j != i]
        if refs:
            scores.append(sentence_bleu(refs, hyp, smoothing_function=smooth))
    return float(np.mean(scores)) if scores else 0.0


# ── Lexical Diversity (TTR) ───────────────────────────────────────────────────

def lexical_diversity(texts: List[str]) -> float:
    """Type-Token Ratio: unique tokens / total tokens across the corpus."""
    all_tokens = " ".join(texts).lower().split()
    return len(set(all_tokens)) / max(len(all_tokens), 1)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python privacy_eval.py <data.pkl>")
        sys.exit(1)

    data = pkl.load(open(sys.argv[1], "rb"))

    synth, ref = [], []
    for i in range(len(data["synth"])):
        if len(data["synth"][i]) > 0:
            synth.append(data["synth"][i])
            ref.append(data["origin"][i])

    print("Eval dataset size is ", len(synth))

    ent_score = entity_recall(synth, ref)
    print(f"Entity-matching recall        : {ent_score:.4f}   (lower better)")

    report = detect_outliers(
        synth, method="distance", radius=0.30, min_neighbors=2, return_embeds=True
    )
    pei = privacy_exposure_index(report["is_outlier"], report["neighbors"], k_thr=3)
    print(f"Privacy-Exposure Index (PEI*) : {pei:.4f}   (lower better)")
    print(f"Min distance to neighbor      : {report['outlier_dist_mean']:.4f}   (lower better)")

    sim = outlier_similarity(ref[:500], synth[:500])
    print(f"Outlier similarity            : {sim:.4f}   (lower better)")

    nndr = nndr_embedding(ref, synth)
    print(f"NNDR Mean                     : {nndr.mean:.3f}  (higher better)")

    out_img = pl.Path(sys.argv[1]).with_suffix(".pca.png")
    plot_pca(report["embeddings"], report["is_outlier"], out_img)
    print(f"PCA saved to {out_img}")
