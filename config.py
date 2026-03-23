# ──────────────────────────────────────────────────────────────────────────────
# config.py  —  STAMP unified configuration
# ──────────────────────────────────────────────────────────────────────────────

# ── Model ──────────────────────────────────────────────────────────────────────
# Set to raw Llama before Stage-1 SFT is run; update to the matching dataset SFT
# path (e.g. "yelp_sft_stage1") before running Stage-2 RL training.
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
DATASET_NAME = "yelp"          # yelp | tweet | imdb | synpai
TEXT_FIELD = "review"
USE_REASONING = False
EXPERIMENT_ID = "stamp_yelp"

# ── Architecture ───────────────────────────────────────────────────────────────
# max_seq_length: total context window for vLLM KV cache and SFTTrainer.
# 2048 comfortably covers prompt (≤768) + completion (≤384) on all datasets.
max_seq_length = 2048
lora_rank = 32                 # LoRA rank; paper uses 32

# max_prompt_length: GRPO hard-truncates prompts to this length.
# Prompts (system + user + style ref + source text) measured at ~536 tokens;
# 768 gives headroom for longer reviews without wasting completion budget.
max_prompt_length = 768

# max_completion_length: explicitly capped rather than derived from
# max_seq_length - max_prompt_length to control GRPO backward memory:
#   GRPO logits peak: num_generations × max_completion_length × vocab(128256) × fp32
#   With num_generations=2 and 384 tokens: 2 × 384 × 128256 × 4B ≈ 0.39 GB
# Yelp/Twitter/IMDb rewrites are rarely longer than 350 tokens.
max_completion_length = 384

# ── GlobalStyleMemory (paper §3.3.1, Eq. 1) ───────────────────────────────────
GSM_LAMBDA = 2.0               # τ = μ + λ·σ  (paper default λ = 2.0)
GSM_MERGE_RADIUS = 0.15        # δ: cosine-distance threshold to merge into existing cluster
GSM_LOCAL_RADIUS = 0.25        # cosine-distance radius for local density s_close
GSM_MAX_NODES = 5000           # cap on number of cluster nodes to prevent OOM
# Quality gate: only add a rewrite to memory if its combined reward exceeds this
# threshold. Prevents early-training low-quality rewrites from polluting the
# style distribution. Set to -inf to disable (add every step unconditionally).
GSM_MIN_REWARD_TO_UPDATE = 0.0

# ── Reward: semantic fidelity (paper §3.3.3) ──────────────────────────────────
# BERT_SCORE_FLOOR: practical normalisation floor mapping [floor, 1.0] → [0, 1].
# Raw English BERTScore rarely falls below 0.80; not explicitly stated in paper.
BERT_SCORE_FLOOR = 0.80

# ── Stage-1 SFT corpus generation ─────────────────────────────────────────────
SEM_THRESHOLD = 0.85           # min BERTScore F1 for a candidate to be accepted
TEACHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"   # or local checkpoint path

# ── Adaptive reward weights (paper §3.4) ──────────────────────────────────────
# The privacy-utility tradeoff differs between outlier and in-distribution
# sources. Outlier sources carry higher stylistic re-identification risk, so
# privacy rewards (entity, GSM, entropy) are up-weighted when the source is
# detected as an outlier; the utility reward (BERTScore) is correspondingly
# scaled down to maintain bounded total reward magnitude.
#
# Combined reward (per sample):
#   r_total = (w_p(x) · p_hist) · [r_entity + r_GSM + r_entropy + r_infer]
#           + (w_u(x) · u_hist) · r_sem
#
#   r_infer = 0 when INFER_REWARD_ENABLED=False (default).
#
#   w_p(x) = W_PRIVACY_BASE + W_OUTLIER_PRIVACY_BOOST   if x ∈ O
#           = W_PRIVACY_BASE                              otherwise
#
#   w_u(x) = W_UTILITY_BASE * W_OUTLIER_UTILITY_SCALE   if x ∈ O
#           = W_UTILITY_BASE                              otherwise
#
# Default values: privacy rewards boosted by +50% and utility reward scaled to
# 80% for outlier sources, keeping total reward in the same range as uniform
# weighting (W_PRIVACY_BASE=1, W_UTILITY_BASE=1, boost=0.5, scale=0.8).
W_PRIVACY_BASE = 1.0
W_UTILITY_BASE = 1.0
W_OUTLIER_PRIVACY_BOOST = 0.5   # added to W_PRIVACY_BASE when source ∈ O
W_OUTLIER_UTILITY_SCALE = 0.8   # multiplied with W_UTILITY_BASE when source ∈ O

# ── History-conditioned reward weights (threshold-based) ───────────────────────
# Each text has its own (privacy, utility) history stored in GlobalStyleMemory.
# The reward weights for the next encounter are adjusted by threshold rules:
#
#   Privacy rule — boost when deficient, hold when sufficient:
#     p_mult = 1 + stale_factor × HISTORY_PRIVACY_BOOST   if past_privacy < PRIVACY_LOW_THRESHOLD
#              1.0                                          otherwise
#
#   Utility rule — relax when already good, hold when not yet sufficient:
#     u_mult = max(HISTORY_UTILITY_MIN,
#                  1 − stale_factor × HISTORY_UTILITY_DISCOUNT)  if past_utility > UTILITY_HIGH_THRESHOLD
#              1.0                                                  otherwise
#
# Staleness schedule: stale_factor = 1 − exp(−Δsteps / HISTORY_DECAY_STEPS)
#   → 0  when freshly rewritten (gradient still propagating; no adjustment yet)
#   → 1  when stale (text needs renewed targeted pressure)
HISTORY_DECAY_STEPS = 200         # staleness half-life in training steps
PRIVACY_LOW_THRESHOLD = 0.4       # below → privacy needs more work → boost p_mult
UTILITY_HIGH_THRESHOLD = 0.7      # above → utility is sufficient → reduce u_mult
HISTORY_PRIVACY_BOOST = 0.5       # max additive boost to p_mult  (stale + deficient)
HISTORY_UTILITY_DISCOUNT = 0.3    # max subtractive discount to u_mult (stale + sufficient)
HISTORY_UTILITY_MIN = 0.5         # floor for u_mult — utility pressure never fully removed

# ── LLM Judger ─────────────────────────────────────────────────────────────────
# When enabled, an LLM is prompted to return structured (privacy, utility) scores
# for each (source, rewrite) pair. This replaces the proxy scores derived from
# r_entity and r_sem. Only enable on a machine with spare GPU memory or when
# running a separate judger process (adds ~200 ms per call).
LLM_JUDGE_ENABLED = False
LLM_JUDGE_MODEL = BASE_MODEL    # reuse the student model as judge by default

# ── Attribute inference threat model (paper §3.3.3) ───────────────────────────
# Zero-shot NLI classifier that estimates the probability that sensitive
# attributes (health status, political views, sexual orientation, etc.) can be
# inferred from the rewritten text.
#
# r_infer(y) = −Σ P(attr_k | y) · sensitivity_k / Σ sensitivity_k   ∈ [−1, 0]
# Higher (closer to 0) = fewer inferable attributes = better implicit privacy.
#
# ZS_CLASSIFIER_MODEL: any HuggingFace NLI model compatible with
#     transformers.pipeline("zero-shot-classification").
#     Default: cross-encoder/nli-deberta-v3-small (~70 MB fp32) — low memory.
#     Higher accuracy option: facebook/bart-large-mnli (~1.6 GB fp16) — needs
#     spare VRAM; not recommended for 8 GB single-GPU setups.
#
# INFER_REWARD_ENABLED: set True to include r_infer in the combined reward.
#     Adds ~50–200 ms per batch depending on model choice and batch size.
#     Default False preserves baseline memory budget; enable when VRAM allows.
ZS_CLASSIFIER_MODEL = "cross-encoder/nli-deberta-v3-small"
INFER_HYPOTHESIS_TEMPLATE = "This text reveals that the author is {}."
INFER_REWARD_ENABLED = False

# ── Ablation flags ─────────────────────────────────────────────────────────────
# ABLATION_ENTROPY_ONLY: when True, the combined reward collapses to
#   r_total = W_PRIVACY_BASE × r_entropy only.
#   Set to True only for ablation experiments (Table 2, row "+RL w/o SMM").
ABLATION_ENTROPY_ONLY = False
