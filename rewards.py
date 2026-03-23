import numpy as np
import torch
from typing import List, Any

from prompt_template import (
    match_format,
    reasoning_start,
    reasoning_end,
    solution_start,
    solution_end,
)
from common import extract_entities, bert_scorer, author_model
from utils import _STYLE_EMBEDDER
from sklearn.metrics.pairwise import cosine_distances as _cosine_distances
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction as _SF
from config import (
    BERT_SCORE_FLOOR,
    W_PRIVACY_BASE,
    W_UTILITY_BASE,
    W_OUTLIER_PRIVACY_BOOST,
    W_OUTLIER_UTILITY_SCALE,
    HISTORY_DECAY_STEPS,
    PRIVACY_LOW_THRESHOLD,
    UTILITY_HIGH_THRESHOLD,
    HISTORY_PRIVACY_BOOST,
    HISTORY_UTILITY_DISCOUNT,
    HISTORY_UTILITY_MIN,
    INFER_REWARD_ENABLED,
    ABLATION_ENTROPY_ONLY,
)
from author import (
    tokenizer as author_tokenizer,
    max_length,
    load_pretrained_author_classifier,
    static_trainers,
    author_id_field,
)


# ── Response extraction helpers ───────────────────────────────────────────────

def _extract_responses(completions: List[Any]) -> List[str]:
    """Robustly extract text from completions of various formats."""
    out = []
    for c in completions:
        if isinstance(c, list) and c and isinstance(c[0], dict) and "content" in c[0]:
            out.append(c[0]["content"])
        elif isinstance(c, dict) and "content" in c:
            out.append(c["content"])
        else:
            out.append(str(c))
    return out


# ── Format rewards (reasoning mode only) ─────────────────────────────────────

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores


# ── Privacy reward: entity overlap ────────────────────────────────────────────

def entity_nominal_value_func(prompts, completions, answer, **kwargs) -> list:
    """
    Explicit PII privacy reward (paper §3.3.3):
        r_entity(y) = -|NER(x) ∩ NER(y)| / (|NER(x)| + ε)
    Normalised to [-1, 0]; higher (closer to 0) = better privacy.
    """
    responses = _extract_responses(completions)
    res = []
    for r, q in zip(responses, answer):
        ner_q = extract_entities(q)
        overlap = ner_q & extract_entities(r)
        res.append(-len(overlap) / (len(ner_q) + 1e-6))
    return res


# ── Semantic fidelity reward: BERTScore ───────────────────────────────────────

def reward_bert_func(prompts, completions, answer, **kwargs) -> list:
    """
    Semantic fidelity reward (paper §3.3.3):
        r_sem(y) = max(0, (BERTScore_F1(x,y) - floor) / (1 - floor))
    Normalised to [0, 1]. floor = BERT_SCORE_FLOOR (default 0.80).
    Paper does not cap the score from above.
    """
    responses = _extract_responses(completions)
    _, _, F1_scores = bert_scorer.score(responses, answer)
    res = [
        float(max(0.0, (float(s) - BERT_SCORE_FLOOR) / (1.0 - BERT_SCORE_FLOOR)))
        for s in F1_scores
    ]
    return res


# ── Style-Aware Privacy-Diversity Tradeoff reward (GlobalStyleMemory) ─────────

def _batch_style_similarity_penalty(responses: list) -> list:
    """
    d_sty: per-response mean cosine SIMILARITY to other responses in the batch.
    High similarity → all rewrites are stylistically converging → penalise.
    Paper §3.3.2 "Global Deviation": penalises homogeneous generation batches.
    """
    if len(responses) <= 1:
        return [0.0]
    embs = np.asarray(
        _STYLE_EMBEDDER.encode(
            responses, convert_to_numpy=True, show_progress_bar=False
        )
    )
    dist_mat = _cosine_distances(embs)
    sim_mat = 1.0 - dist_mat
    np.fill_diagonal(sim_mat, 0.0)
    # Mean similarity to OTHER responses (self excluded via zero diagonal)
    d_sty = sim_mat.sum(axis=1) / max(len(responses) - 1, 1)
    return d_sty.tolist()


def _batch_self_bleu(responses: list) -> list:
    """
    d_bleu: per-response BLEU against all other responses in the batch.
    High BLEU → lexically repetitive batch → penalise.
    Paper §3.3.2: diversity penalty component.
    """
    smooth = _SF().method1
    tokenised = [r.split() for r in responses]
    scores = []
    for i, hyp in enumerate(tokenised):
        refs = [t for j, t in enumerate(tokenised) if j != i]
        if not refs:
            scores.append(0.0)
        else:
            scores.append(sentence_bleu(refs, hyp, smoothing_function=smooth))
    return scores


def reward_gsm_func(prompts, completions, answer, **kwargs) -> list:
    """
    Style-Aware Privacy-Diversity Tradeoff reward (paper §3.3.3, Eq. 2):

        r_GSM(y) = {
            s_avg - (d_sty + d_bleu)                  if x ∈ O  (source is outlier)
            (s_avg + s_close) - (d_sty + d_bleu)      otherwise
        }

    s_avg   = global_similarity(y)   — how close the REWRITE is to mainstream style
    s_close = local_density(y)       — local neighbourhood density of the rewrite
    d_sty   = batch style similarity — penalty: homogeneous batch styles
    d_bleu  = self-BLEU              — penalty: lexically repetitive batch

    Outlier branch: outlier SOURCES need aggressive style normalisation, so we
    reward global mainstream alignment (s_avg) without the local density bonus.
    s_close is dropped because proximity to the source's unusual style region is
    irrelevant — we want the rewrite fully integrated into the mainstream.

    Non-outlier branch: reward stylistic integration (s_avg) plus local density
    (s_close), penalising a homogeneous generation batch.

    NOTE: outlier status is determined from the SOURCE text (answer), not the
    rewrite, because the branching logic depends on the original document's style.

    Reads style_memory from common module (set by train.py at runtime).
    Returns [0.0, ...] if memory has not been initialised yet.
    """
    from common import style_memory
    if style_memory is None:
        return [0.0] * len(completions)

    responses = _extract_responses(completions)

    # Fast path: use pre-computed scores passed by adaptive_combined_reward_func.
    # _gsm_resp: batch_score(responses) — supplies s_avg, s_close per rewrite.
    # _gsm_src : batch_score(sources)  — supplies is_outlier per source.
    # This eliminates 3× per-sample _embed() calls when called from the
    # combined reward (which already batched both sets in two matrix ops).
    gsm_resp = kwargs.get("_gsm_resp")
    gsm_src  = kwargs.get("_gsm_src")

    if gsm_resp is not None and gsm_src is not None:
        s_avg_list   = [s["s_avg"]      for s in gsm_resp]
        s_close_list = [s["s_close"]    for s in gsm_resp]
        is_out_list  = [s["is_outlier"] for s in gsm_src]
    else:
        # Standalone path: called without pre-computed data.
        s_avg_list, s_close_list, is_out_list = [], [], []
        for r, src in zip(responses, answer):
            is_out, _ = style_memory.is_outlier(src)
            s_avg_list.append(style_memory.global_similarity(r))
            s_close_list.append(style_memory.local_density(r))
            is_out_list.append(is_out)

    d_sty_list = _batch_style_similarity_penalty(responses)
    d_bleu_list = _batch_self_bleu(responses)

    rewards = []
    for s_avg, s_close, d_sty, d_bleu, is_out in zip(
        s_avg_list, s_close_list, d_sty_list, d_bleu_list, is_out_list
    ):
        if is_out:
            # Source is stylistic outlier: push rewrite hard toward mainstream.
            # Drop s_close — local density near the outlier's style region is not
            # a meaningful reward signal.
            reward = s_avg - (d_sty + d_bleu)
        else:
            # In-distribution source: reward global + local style alignment while
            # penalising a homogeneous generation batch.
            reward = (s_avg + s_close) - (d_sty + d_bleu)
        rewards.append(reward)

    return rewards


# ── Author classification entropy reward ──────────────────────────────────────

def author_classification_entropy_func(
    prompts, completions, answer, **kwargs
) -> list:
    """
    Implicit privacy reward (paper §3.3.3):
        r_entropy(y) = H_φ(y) / log(n_classes)
    where H_φ(y) = -Σ p_φ(a|y) log p_φ(a|y).
    Higher entropy → more uncertain author attribution → better privacy.
    Normalised to [0, 1] by maximum possible entropy log(n_classes).
    """
    responses = _extract_responses(completions)

    if author_model is None:
        return [0.0] * len(responses)

    try:
        tokenized = author_tokenizer(
            responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            logits = author_model(**tokenized).logits
            probs = torch.softmax(logits, dim=-1)
            entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            n_classes = logits.shape[-1]
            max_entropy = np.log(n_classes)
            normalised = entropies / max_entropy
            return normalised.cpu().numpy().tolist()
    except Exception as e:
        print(f"Warning: author_classification_entropy_func failed: {e}")
        return [0.0] * len(responses)


# ── Attribute inference penalty: implicit privacy (new contribution) ──────────

def reward_infer_func(prompts, completions, answer, **kwargs) -> list:
    """
    Implicit privacy reward: attribute inference penalty (new contribution §3.3.3).

        r_infer(y) = −Σ P(attr_k | y) · sensitivity_k / Σ sensitivity_k

    Normalised to [−1, 0]; higher (closer to 0) = fewer inferable attributes =
    better implicit privacy protection.

    Uses a zero-shot NLI classifier (PrivacyThreatModel) to estimate the
    probability that each sensitive attribute (health status, sexual orientation,
    political views, etc.) can be inferred from the rewrite. The weighted sum
    reflects GDPR Article 9 sensitivity ordering.

    Returns [0.0, ...] when INFER_REWARD_ENABLED=False (config default) so the
    function can be called unconditionally without loading the NLI model.
    """
    responses = _extract_responses(completions)
    if not INFER_REWARD_ENABLED:
        return [0.0] * len(responses)

    from privacy_threat_model import get_threat_model
    graphs = get_threat_model().build_graphs(responses, per_span=False)
    return [-g.risk_score for g in graphs]


# ── Adaptive combined reward (new contribution) ────────────────────────────────

def _history_adjustment(
    history, current_step: int
) -> tuple:
    """
    Compute (p_mult, u_mult) from a node's HistoryEntry using threshold-based rules.

    Staleness schedule
    ------------------
    stale_factor = 1 − exp(−Δsteps / HISTORY_DECAY_STEPS)
      → 0  immediately after rewriting (gradient still propagating; hold adjustment)
      → 1  when the node is stale (text needs fresh targeted pressure)

    Privacy rule — boost when deficient, hold when sufficient
    ---------------------------------------------------------
    if past_privacy < PRIVACY_LOW_THRESHOLD:
        p_mult = 1 + stale_factor × HISTORY_PRIVACY_BOOST
                 # e.g. score=0.2, stale → 1 + 1.0×0.5 = 1.50
    else:
        p_mult = 1.0
                 # privacy is acceptable; extra weight risks harming utility

    Utility rule — relax when already good, hold when still needed
    -------------------------------------------------------------
    if past_utility > UTILITY_HIGH_THRESHOLD:
        u_mult = max(HISTORY_UTILITY_MIN, 1 − stale_factor × HISTORY_UTILITY_DISCOUNT)
                 # e.g. score=0.85, stale → max(0.5, 1 − 0.3) = 0.70
    else:
        u_mult = 1.0
                 # utility not yet sufficient; maintain full pressure

    The asymmetry is intentional:
      - Privacy needs explicit remediation until it clears the floor.
      - Utility needs deliberate relaxation once it is good enough, to free
        gradient budget for privacy improvement.

    Returns (1.0, 1.0) when no history is available (first encounter).
    """
    if history is None:
        return 1.0, 1.0

    staleness = max(0, current_step - history.step)
    stale_factor = 1.0 - np.exp(-staleness / max(HISTORY_DECAY_STEPS, 1))

    # Privacy: boost when below threshold, neutral otherwise.
    if history.privacy_score < PRIVACY_LOW_THRESHOLD:
        p_mult = 1.0 + stale_factor * HISTORY_PRIVACY_BOOST
    else:
        p_mult = 1.0

    # Utility: discount when above threshold (already sufficient), neutral otherwise.
    if history.utility_score > UTILITY_HIGH_THRESHOLD:
        u_mult = max(HISTORY_UTILITY_MIN, 1.0 - stale_factor * HISTORY_UTILITY_DISCOUNT)
    else:
        u_mult = 1.0

    return float(p_mult), float(u_mult)


def adaptive_combined_reward_func(
    prompts, completions, answer, **kwargs
) -> list:
    """
    History-aware, outlier-adaptive privacy-utility weighted reward.

    Combines two adaptive mechanisms:

    1. Outlier-adaptive weighting (§ new contribution A):
           w_p(x) = W_PRIVACY_BASE + W_OUTLIER_PRIVACY_BOOST   if x ∈ O
           w_u(x) = W_UTILITY_BASE × W_OUTLIER_UTILITY_SCALE   if x ∈ O

    2. History-conditioned multiplier (§ new contribution B):
           p_hist_mult = 1 + stale × deficit_privacy × 2 × HISTORY_PRIVACY_SCALE
           u_hist_mult = 1 + stale × deficit_utility × 2 × HISTORY_UTILITY_SCALE

    Combined reward:
        r_total = (w_p × p_hist_mult) × [r_entity + r_GSM + r_entropy + r_infer]
                + (w_u × u_hist_mult) × r_sem

    r_infer is the attribute inference penalty (new contribution §3.3.3):
        r_infer(y) = −Σ P(attr_k | y) · sensitivity_k / Σ sensitivity_k  ∈ [−1, 0]
    Active only when INFER_REWARD_ENABLED=True in config.py (default: False).

    Side effect:
        Writes per-sample {"privacy", "utility"} proxy scores to
        common._last_step_component_scores so that MemoryUpdateCallback
        can update node history without re-running reward computation.
    """
    import common
    from common import style_memory

    # ── Batch-compute all GSM signals: 2 embedding calls total ────────────────
    # Calling is_outlier / global_similarity / local_density / get_node_history
    # individually embeds the same text 3–4 times. batch_score() does one embed
    # per group (responses + sources), covering all signals in two matrix ops.
    responses = _extract_responses(completions)
    gsm_resp = style_memory.batch_score(responses) if style_memory is not None else None
    gsm_src  = style_memory.batch_score(list(answer)) if style_memory is not None else None

    # ── Compute all component rewards ──────────────────────────────────────────
    r_entity_list = entity_nominal_value_func(prompts, completions, answer, **kwargs)
    r_sem_list    = reward_bert_func(prompts, completions, answer, **kwargs)
    r_gsm_list    = reward_gsm_func(
        prompts, completions, answer,
        _gsm_resp=gsm_resp, _gsm_src=gsm_src,
        **kwargs,
    )
    r_entropy_list = author_classification_entropy_func(
        prompts, completions, answer, **kwargs
    )
    r_infer_list  = reward_infer_func(prompts, completions, answer, **kwargs)

    # ── Per-source: outlier status + node history (from pre-computed batch) ───
    # nearest_idx from gsm_src gives O(1) history lookup — no extra embed call.
    is_out_list, history_list = [], []
    current_step = kwargs.get("current_step", common._current_step)

    if gsm_src is not None:
        for sc, src_text in zip(gsm_src, answer):
            is_out_list.append(sc["is_outlier"])
            # Text-level history: O(1) exact-string lookup, independent per source.
            history_list.append(style_memory.get_text_history(src_text))
    else:
        for _ in answer:
            is_out_list.append(False)
            history_list.append(None)

    # ── Compute adaptive weighted sum + build full component bus ──────────────
    rewards = []
    component_scores = []
    for r_e, r_s, r_g, r_h, r_i, is_out, hist in zip(
        r_entity_list, r_sem_list, r_gsm_list, r_entropy_list, r_infer_list,
        is_out_list, history_list,
    ):
        # Outlier-adaptive base weights
        if is_out:
            w_p = W_PRIVACY_BASE + W_OUTLIER_PRIVACY_BOOST
            w_u = W_UTILITY_BASE * W_OUTLIER_UTILITY_SCALE
        else:
            w_p = W_PRIVACY_BASE
            w_u = W_UTILITY_BASE

        # History-conditioned multipliers
        p_hist, u_hist = _history_adjustment(hist, current_step)

        if ABLATION_ENTROPY_ONLY:
            # Ablation: only entropy, no GSM direction, no utility, no entity
            r_total = W_PRIVACY_BASE * r_h
        else:
            r_total = (w_p * p_hist) * (r_e + r_g + r_h + r_i) + (w_u * u_hist) * r_s
        rewards.append(r_total)

        # Full component bus — read by RewardComponentLogger and MemoryUpdateCallback
        component_scores.append({
            # Proxy scores for MemoryUpdateCallback / history
            "privacy":    float(np.clip(1.0 + r_e, 0.0, 1.0)),
            "utility":    float(np.clip(r_s, 0.0, 1.0)),
            # Individual reward components — for convergence logging / ablation analysis
            "r_entity":   float(r_e),
            "r_gsm":      float(r_g),
            "r_entropy":  float(r_h),
            "r_sem":      float(r_s),
            "r_infer":    float(r_i),
            "r_total":    float(r_total),
            "is_outlier": bool(is_out),
        })

    common._last_step_component_scores = component_scores
    return rewards
