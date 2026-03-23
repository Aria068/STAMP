import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from config import GSM_LOCAL_RADIUS, GSM_MERGE_RADIUS


@dataclass
class HistoryEntry:
    """
    Records the outcome of the last rewrite attempt for a memory node.

    Stored per memory node so that reward weighting can be conditioned on
    past performance. Updated by MemoryUpdateCallback after each RL step.

    Fields
    ------
    step          : global training step when this node was last rewritten.
    privacy_score : privacy quality of the best rewrite, ∈ [0, 1].
                    1 = perfect privacy (no entities leaked).
                    Derived from r_entity (proxy) or LLM judger.
    utility_score : semantic fidelity of the best rewrite, ∈ [0, 1].
                    1 = perfect utility (meaning fully preserved).
                    Derived from r_sem (proxy) or LLM judger.
    rewrite_text  : the best rewrite text that produced these scores.
    """
    step: int
    privacy_score: float
    utility_score: float
    rewrite_text: str

# Single module-level embedder instance shared across all classes.
# Exported as _STYLE_EMBEDDER so sft_data_gen.py and rewards.py can import it
# directly without loading a second copy of the 400 MB model.
_STYLE_EMBEDDER = SentenceTransformer("AnnaWegmann/Style-Embedding")


def sample_load_save(csv_path, sample_num, dataset_name, random=False):
    """
    Sample data from a CSV file and save to a new file.
    """
    import os

    data = pd.read_csv(csv_path)
    sample_num = min(sample_num, len(data))

    if random:
        sampled_data = data.sample(n=sample_num, random_state=42)
    else:
        sampled_data = data.iloc[:sample_num]

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{dataset_name}_review_{sample_num}.csv"
    sampled_data.to_csv(output_path, index=False)
    return output_path


class GlobalStyleMemory:
    """
    Cluster-based style memory bank (Algorithm 1, paper §3.3.1).

    Paper name: "Global Style Memory module" (GSM, §3.1).

    During construction (fit) and online update (add_node), each new embedding
    is either merged into the nearest existing cluster (if cosine distance < δ)
    or spawns a new cluster node.  Merging updates the cluster centroid via an
    incremental weighted mean and increments its visit weight.

    Used for:
      - Per-sample outlier detection:  d_x > τ  (paper §3.3.1, Eq. 1)
      - Adaptive reference sampling:  common style for outliers,
                                      diverse style for in-distribution inputs
      - Reward signals: s_avg (global similarity), s_close (local density)
      - Self-evolving update: add best rewrite per RL step (paper §3.3.2)
    """

    def __init__(
        self,
        lambda_sensitivity: float = 2.0,
        merge_radius: float = GSM_MERGE_RADIUS,
    ):
        self.lambda_ = lambda_sensitivity        # τ = μ + λ·σ (paper default 2.0)
        self._delta = merge_radius               # δ: cluster merge radius
        self._embeddings: list = []              # np.ndarray (dim,) — cluster centroid
        self._texts: list = []                   # representative text per cluster
        self._weights: list = []                 # accumulated visit weight per cluster
        self._tau: Optional[float] = None        # outlier threshold
        self._mu: Optional[float] = None
        self._sigma: Optional[float] = None
        self._n_init: int = 0                    # K after fit(); frozen denominator for s_close
        # Text-level history: one HistoryEntry per source text (keyed by exact string).
        # O(1) lookup; independent across texts that share the same cluster.
        self._text_history: dict = {}            # str → HistoryEntry

    # ── Embedding ─────────────────────────────────────────────────────────────
    def _embed(self, texts: list) -> np.ndarray:
        return np.asarray(
            _STYLE_EMBEDDER.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
        )

    # ── Build ─────────────────────────────────────────────────────────────────
    def fit(self, texts: list):
        """
        Initialise from training corpus using cluster-based merging (Algorithm 1).

        For each sentence x in D:
          - Compute style embedding e = f_style(x).
          - Find nearest existing cluster centroid i* (min cosine distance).
          - If d_cos(e, e_{i*}) < δ: merge into i* (update centroid + weight).
          - Else: create a new cluster node with centroid e and weight 1.
        Finally recompute the outlier threshold τ.
        """
        embs = self._embed(texts)          # single batch embedding call
        self._embeddings = []
        self._texts = []
        self._weights = []
        self._text_history = {}            # reset per-text history on re-fit

        for text, emb in zip(texts, embs):
            if not self._embeddings:
                self._embeddings.append(emb.copy())
                self._texts.append(text)
                self._weights.append(1.0)
                continue

            X = np.vstack(self._embeddings)
            dists = cosine_distances(emb.reshape(1, -1), X).flatten()
            i_star = int(np.argmin(dists))

            if dists[i_star] < self._delta:
                # Merge: incremental centroid update  e_new = (e_old*w + e) / (w+1)
                w = self._weights[i_star]
                self._embeddings[i_star] = (self._embeddings[i_star] * w + emb) / (w + 1.0)
                self._weights[i_star] = w + 1.0
            else:
                # New cluster node
                self._embeddings.append(emb.copy())
                self._texts.append(text)
                self._weights.append(1.0)

        self._recompute_threshold()
        self._n_init = len(self._embeddings)     # freeze denominator for s_close

    def _recompute_threshold(self):
        """
        τ = μ + λ·σ  over per-cluster-centroid mean pairwise cosine distances.
        Paper §3.3.1, Eq. 1 / Algorithm 1 line 16.
        """
        if len(self._embeddings) < 2:
            # Single cluster: set a neutral threshold (everything is in-distribution).
            self._mu = 0.5
            self._sigma = 0.0
            self._tau = 0.5
            return
        X = np.vstack(self._embeddings)
        d = cosine_distances(X)
        np.fill_diagonal(d, np.nan)
        mean_d = np.nanmean(d, axis=1)
        self._mu = float(np.nanmean(mean_d))
        self._sigma = float(np.nanstd(mean_d)) + 1e-9
        self._tau = self._mu + self.lambda_ * self._sigma

    # ── Outlier detection ──────────────────────────────────────────────────────
    def style_distance(self, text: str) -> float:
        """Mean cosine distance from text to all memory nodes (d_x in paper)."""
        v = self._embed([text])
        X = np.vstack(self._embeddings)
        return float(np.mean(cosine_distances(v, X).flatten()))

    def is_outlier(self, text: str) -> tuple:
        """
        Return (is_outlier: bool, d_x: float).
        Outlier when d_x > τ  (paper §3.3.1).
        """
        d_x = self.style_distance(text)
        return d_x > self._tau, d_x

    # ── Reference sampling ─────────────────────────────────────────────────────
    def sample_reference_common(self) -> str:
        """
        For OUTLIER inputs: return the text at the node with the highest
        accumulated weight (most-visited = most common / neutral style).

        Paper §3.3.1: "reference is selected from the style node with the
        highest weight (i.e., the most common style)."
        """
        idx = int(np.argmax(self._weights))
        return self._texts[idx]

    def sample_reference_diverse(self, text: str) -> str:
        """
        For IN-DISTRIBUTION inputs: return the memory node that is most
        stylistically distant from the source, restricted to nodes that are
        themselves in-distribution (d_node ≤ τ).

        Paper §3.3.1: "reference is chosen from a nearby style cluster that
        maximises stylistic diversity while maintaining contextual relevance."

        The in-distribution constraint ensures the reference is a 'normal'
        style (contextually plausible) while being maximally diverse from
        the source's style.
        """
        v = self._embed([text])
        X = np.vstack(self._embeddings)
        dists_to_source = cosine_distances(v, X).flatten()

        # Identify in-distribution nodes (mean pairwise distance ≤ τ)
        pairwise = cosine_distances(X)
        np.fill_diagonal(pairwise, np.nan)
        node_mean_dists = np.nanmean(pairwise, axis=1)
        in_dist_mask = node_mean_dists <= self._tau

        if not np.any(in_dist_mask):
            # Fallback: all nodes are outliers; pick most distant regardless
            return self._texts[int(np.argmax(dists_to_source))]

        # Among in-distribution nodes, pick the one farthest from the source
        candidate_dists = np.where(in_dist_mask, dists_to_source, -np.inf)
        idx = int(np.argmax(candidate_dists))
        return self._texts[idx]

    # ── Reward signals ─────────────────────────────────────────────────────────
    def global_similarity(self, text: str) -> float:
        """
        s_avg: average cosine SIMILARITY to all memory nodes (1 - mean distance).
        High s_avg means the rewrite's style aligns with the population —
        desirable for in-distribution rewrites, penalised for outliers.
        """
        return 1.0 - self.style_distance(text)

    def local_density(self, text: str, radius: float = 0.25) -> float:
        """
        s_close: fraction of initial clusters within cosine-distance `radius`.
        Denominator is frozen at fit() time (_n_init) so the reward scale is
        stationary throughout training regardless of how many new clusters
        are added by add_node().
        """
        v = self._embed([text])
        X = np.vstack(self._embeddings)
        dists = cosine_distances(v, X).flatten()
        return float(np.sum(dists < radius)) / max(self._n_init, 1)

    # ── Self-evolving update ───────────────────────────────────────────────────
    def add_node(self, text: str, weight: float = 1.0, max_nodes: int = 5000):
        """
        Online update after each RL step (paper §3.3.2 / Algorithm 1 online block).

        Mirrors the fit() merge-or-create logic:
          - If nearest cluster centroid is within δ: merge (update centroid + weight).
          - Else: create a new cluster node (if pool is below max_nodes cap).
        Recomputes τ every 100 calls to amortise the O(n²) pairwise cost.
        """
        emb = self._embed([text])[0]
        X = np.vstack(self._embeddings)
        dists = cosine_distances(emb.reshape(1, -1), X).flatten()
        i_star = int(np.argmin(dists))

        if dists[i_star] < self._delta:
            # Merge into existing cluster: incremental centroid + weight update.
            w = self._weights[i_star]
            self._embeddings[i_star] = (self._embeddings[i_star] * w + emb) / (w + 1.0)
            self._weights[i_star] = w + weight
        elif len(self._embeddings) < max_nodes:
            # New cluster node.
            self._embeddings.append(emb)
            self._texts.append(text)
            self._weights.append(weight)

        if len(self._embeddings) % 100 == 0:
            self._recompute_threshold()

    # ── Batch scoring (single embedding call for a list of texts) ─────────────

    def batch_score(
        self, texts: list, radius: Optional[float] = None
    ) -> list:
        """
        Compute all GSM signals for a batch of texts in **two** matrix operations:
          1. One _embed() call → V  (n_texts, dim)
          2. One cosine_distances(V, X) call → D  (n_texts, n_nodes)

        Returns one dict per text:
            is_outlier  : bool   — d_x > τ
            d_x         : float  — mean cosine distance to cluster centroids
            s_avg       : float  — 1 - d_x  (global style similarity)
            s_close     : float  — fraction of clusters within `radius`
            nearest_idx : int    — argmin of D[i], nearest cluster index

        Compared with calling is_outlier / global_similarity / local_density
        individually per text (each re-embeds), this reduces SentenceTransformer
        calls from O(3 × n_texts) to O(1) per batch.
        """
        if not texts:
            return []
        if radius is None:
            radius = GSM_LOCAL_RADIUS

        V = self._embed(texts)                    # (n_texts, dim)
        X = np.vstack(self._embeddings)           # (n_clusters, dim)
        D = cosine_distances(V, X)                # (n_texts, n_clusters)
        n_init = max(self._n_init, 1)             # frozen denominator — stationary scale

        results = []
        for d_row in D:
            d_x = float(np.mean(d_row))
            results.append({
                "is_outlier": d_x > self._tau,
                "d_x":         d_x,
                "s_avg":       1.0 - d_x,
                "s_close":     float(np.sum(d_row < radius)) / n_init,
                "nearest_idx": int(np.argmin(d_row)),
            })
        return results

    # ── Text-level history access ──────────────────────────────────────────────

    def get_text_history(self, text: str) -> Optional[HistoryEntry]:
        """
        Return the HistoryEntry for this exact source text, or None if it has
        never been rewritten. O(1) dict lookup — no embedding call required.

        Each source text has an independent entry: texts that share a cluster
        centroid do NOT share history, giving finer-grained reward conditioning.
        """
        return self._text_history.get(text)

    def update_node_history(
        self,
        source: str,
        step: int,
        privacy_score: float,
        utility_score: float,
        rewrite_text: str,
    ) -> None:
        """
        Record the outcome of the latest rewrite attempt for this source text.
        Called by MemoryUpdateCallback after each RL step.

        Stored in _text_history keyed by the exact source string, so every
        distinct source text accumulates its own independent privacy/utility
        trajectory regardless of which cluster it belongs to.

        Parameters
        ----------
        source        : original source text (dict key)
        step          : current global training step
        privacy_score : ∈ [0,1], 1 = perfect privacy
        utility_score : ∈ [0,1], 1 = perfect utility
        rewrite_text  : the best rewrite that produced these scores
        """
        self._text_history[source] = HistoryEntry(
            step=step,
            privacy_score=float(np.clip(privacy_score, 0.0, 1.0)),
            utility_score=float(np.clip(utility_score, 0.0, 1.0)),
            rewrite_text=rewrite_text,
        )

    @property
    def tau(self) -> float:
        return self._tau
