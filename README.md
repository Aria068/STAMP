# STAMP: Stylometric Text Anonymization with Memory-guided Policy Optimization

Official code release for the paper:

> **STAMP: Stylometric Text Anonymization with Memory-guided Policy Optimization**

STAMP is a two-stage (SFT + GRPO reinforcement learning) framework for authorship anonymization. It uses a **Style Manifold Memory (SMM)** — a cluster-based style density estimator — to adaptively weight privacy and utility rewards, pushing rewrites away from stylistically distinctive regions while preserving semantic content.

---

## Architecture Overview

```
Stage 1a: sft_data_gen.py   →  Teacher LLM generates pseudo-parallel corpus
Stage 1b: sft_train.py      →  SFT fine-tunes student Llama-3.2-3B on corpus
Stage 2:  train.py          →  GRPO RL with self-evolving SMM reward
Inference: generate.py      →  Batch generation with trained LoRA
Eval:      eval_all.sh      →  Privacy + diversity + meaning metrics
```

### Key components

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters and flags |
| `utils.py` | `GlobalStyleMemory` + `HistoryEntry` |
| `train.py` | `SelfEvolvingGRPOTrainer` + `MemoryUpdateCallback` |
| `rewards.py` | All reward functions + `adaptive_combined_reward_func` |
| `common.py` | Shared singletons (BERTScorer, style_memory, author model) |
| `llm_judger.py` | Proxy / LLM judge for (privacy, utility) history scoring |
| `privacy_threat_model.py` | Zero-shot NLI attribute inference (`r_infer`) |
| `dataset.py` | Dataset loaders + memory-aware prompt construction |
| `prompt_template.py` | System prompt + regex patterns |
| `author.py` | BERT-based author classifier (entropy reward) |
| `sft_data_gen.py` | Stage 1a: teacher-guided SFT corpus generation |
| `sft_train.py` | Stage 1b: SFT fine-tuning |
| `generate.py` | Inference / batch generation |
| `privacy_eval.py` | Privacy metrics (entity recall, PEI, NNDR, outlier similarity) |
| `diversity_eval.py` | Diversity metrics (self-BLEU, distinct-n) |
| `meaning_eval.py` | Utility metrics (BERTScore, MAUVE) |
| `eval_all.sh` | Evaluation sweep loop over models × datasets |

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

GPU requirements: 16 GB VRAM recommended for training (tested on A100 40 GB).
For 8 GB GPUs, set `gpu_memory_utilization=0.55` in `config.py`.

---

## Quick Start

### 0. Prepare data

Place dataset CSVs under `clean_data/`. See `data/README.md` for details.

### 1. Pretrain the author classifier

The author entropy reward requires a BERT author classifier pretrained on the clean dataset:

```bash
python - <<'EOF'
from author import pretrain_author_classifier
pretrain_author_classifier(
    raw_data_path="clean_data/yelp_train_split.csv",
    text_field="review",
    label_field="author_id",
)
EOF
```

### 2. Stage 1a — Generate SFT corpus (requires 70B teacher on multi-GPU)

```bash
python sft_data_gen.py \
    --dataset yelp \
    --teacher_model meta-llama/Llama-3.3-70B-Instruct \
    --output clean_data/yelp_sft_pairs.csv \
    --sample 5000 \
    --tensor_parallel 2
```

### 3. Stage 1b — SFT fine-tuning

```bash
python sft_train.py \
    --data clean_data/yelp_sft_pairs.csv \
    --output yelp_sft_stage1 \
    --epochs 3
```

After this completes, update `BASE_MODEL = "yelp_sft_stage1"` in `config.py`.

### 4. Stage 2 — GRPO RL training

```bash
python train.py
```

Training config is controlled entirely via `config.py`. Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `DATASET_NAME` | `"yelp"` | Dataset to train on |
| `BASE_MODEL` | `"unsloth/Llama-3.2-3B-Instruct-bnb-4bit"` | Student model |
| `ABLATION_ENTROPY_ONLY` | `False` | Full STAMP system (set True for ablation) |
| `INFER_REWARD_ENABLED` | `False` | Enable r_infer (needs extra VRAM) |
| `LLM_JUDGE_ENABLED` | `False` | Enable LLM-based history scoring |

### 5. Inference

```bash
python generate.py \
    --model_version stamp_yelp/final_lora_yelp_stamp_yelp \
    --dataset_path clean_data/yelp_test_split.csv \
    --text review
```

Output is saved to `results/<model>_<dataset>_syn.p` (pickle with `origin` and `synth` keys) and `.csv`.

### 6. Evaluation

```bash
# All models × all datasets
bash eval_all.sh

# Single model / dataset
MODELS="ours" DATASETS="yelp" bash eval_all.sh
```

Each run's output file must be at `results/<model>_<dataset>_syn.p`.

---

## Reward Function

The composite reward combines four signals:

```
r_total = (w_p(x) × p_hist) × [r_entity + r_GSM + r_entropy + r_infer]
        + (w_u(x) × u_hist) × r_sem

w_p(x) = W_PRIVACY_BASE (+ W_OUTLIER_PRIVACY_BOOST if x ∈ O)
w_u(x) = W_UTILITY_BASE (* W_OUTLIER_UTILITY_SCALE if x ∈ O)
p_hist = 1 + stale_factor × deficit_privacy × HISTORY_PRIVACY_BOOST
```

| Reward | Range | Description |
|--------|-------|-------------|
| `r_entity` | [−1, 0] | Entity suppression: −\|NER(x) ∩ NER(y)\| / \|NER(x)\| |
| `r_GSM` | varies | Style-manifold reward: global similarity + local density − diversity penalties |
| `r_entropy` | [0, 1] | Author entropy: H(f_auth(y)) / log(n_classes) |
| `r_sem` | [0, 1] | Semantic fidelity: normalised BERTScore F1 |
| `r_infer` | [−1, 0] | Attribute inference penalty (disabled by default) |

---

## Baselines

Baseline scripts are in `baselines/`. Each produces `results/<method>_<dataset>_syn.p`.

| Script | Baseline |
|--------|----------|
| `baselines/run_presidio.py` | Presidio NER redaction |
| `baselines/run_dpmlm.py` | DP-MLM (ε=50) |
| `baselines/run_dipper.py` | DIPPER T5-11B paraphrase |
| `baselines/run_stylemix.py` | StyleMix LoRA adapters |
| `baselines/run_tarot.py` | TAROT-DPO (GPT-2) |

---

## Models

| Role | Model |
|------|-------|
| Student | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (4-bit NF4, LoRA rank-32) |
| Teacher (SFT) | `meta-llama/Llama-3.3-70B-Instruct` (frozen) |
| Style encoder | `AnnaWegmann/Style-Embedding` (frozen) |
| Author classifier | `bert-base-cased` (fine-tuned per dataset) |
| Semantic scorer | `roberta-large` (BERTScore) |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{stamp2025,
  title     = {STAMP: Stylometric Text Anonymization with Memory-guided Policy Optimization},
  year      = {2025},
}
```
