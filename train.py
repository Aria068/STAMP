import json
import random
import numpy as np
import pandas as pd
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers.trainer_callback import TrainerCallback

from dataset import DATASET_PATH, get_dataset, remove_empty_rows
from config import (
    BASE_MODEL,
    DATASET_NAME,
    TEXT_FIELD,
    max_seq_length,
    lora_rank,
    max_prompt_length,
    max_completion_length,
    EXPERIMENT_ID,
    GSM_LAMBDA,
    GSM_MERGE_RADIUS,
    GSM_MAX_NODES,
    GSM_MIN_REWARD_TO_UPDATE,
    ABLATION_ENTROPY_ONLY,
)
from utils import GlobalStyleMemory
import common
from rewards import adaptive_combined_reward_func
from llm_judger import LLMJudger


# ── Reward component logger callback ──────────────────────────────────────────


class RewardComponentLogger(TrainerCallback):
    """
    Appends mean per-step reward components to a JSONL file after every log step.

    Output file: {output_dir}/reward_components.jsonl
    Each line is a JSON object:
        {"step": int, "entropy": float, "r_gsm": float, "r_entity": float,
         "r_sem": float, "r_infer": float, "r_total": float,
         "outlier_frac": float}

    This file is the input for entropy_convergence_plot.py:
        python entropy_convergence_plot.py \
            --run "BeyondPII={output_dir}/reward_components.jsonl" \
            --entropy-key r_entropy
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._fh = open(log_path, "a")

    def on_log(self, args, state, control, logs=None, **kwargs):
        scores = common._last_step_component_scores
        if not scores:
            return
        keys = ["r_entity", "r_gsm", "r_entropy", "r_sem", "r_infer", "r_total"]
        entry = {"step": state.global_step}
        for k in keys:
            vals = [s[k] for s in scores if k in s]
            entry[k] = float(np.mean(vals)) if vals else float("nan")
        entry["outlier_frac"] = float(
            np.mean([s.get("is_outlier", False) for s in scores])
        )
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self._fh.close()


# ── Self-evolving memory update callback ──────────────────────────────────────


class MemoryUpdateCallback(TrainerCallback):
    """
    After each RL step:
      1. Adds the best rewrite to GlobalStyleMemory (self-evolving paper §3.3.2).
      2. Records the (privacy, utility) scores for that source in node history,
         so future reward calls can apply history-conditioned weight adjustments.

    The LLMJudger decides whether to derive scores from proxy reward values
    (default, zero overhead) or from an explicit LLM judge call.
    """

    def __init__(
        self,
        memory: GlobalStyleMemory,
        judger: LLMJudger,
        max_nodes: int = 5000,
        min_reward: float = 0.0,
    ):
        self.memory = memory
        self.judger = judger
        self.max_nodes = max_nodes
        self.min_reward = min_reward

    def on_step_end(self, args, state, control, **kwargs):
        best        = getattr(state, "_gsm_best_completion", None)
        best_reward = getattr(state, "_gsm_best_reward", None)

        # Quality gate: skip early-training rewrites that are still poor.
        # Only add to memory when the combined reward clears min_reward.
        reward_ok = best_reward is None or best_reward >= self.min_reward
        if best and reward_ok:
            self.memory.add_node(best, max_nodes=self.max_nodes)

            # Update node history with (privacy, utility) scores
            source = getattr(state, "_gsm_best_source", None)
            scores = getattr(state, "_gsm_best_scores", None)
            if source and scores:
                privacy, utility = self.judger.score(
                    source=source,
                    rewrite=best,
                    r_entity=scores.get("r_entity", 0.0),
                    r_sem=scores.get("r_sem", 0.5),
                )
                self.memory.update_node_history(
                    source=source,
                    step=state.global_step,
                    privacy_score=privacy,
                    utility_score=utility,
                    rewrite_text=best,
                )

            # Clear per-step state
            state._gsm_best_completion = None
            state._gsm_best_source = None
            state._gsm_best_scores = None
            state._gsm_best_reward = None
        return control


# ── Extended GRPO trainer: surfaces best completion for memory update ──────────


class SelfEvolvingGRPOTrainer(GRPOTrainer):
    """
    Extends GRPOTrainer to:
      - Record the best-reward completion per step for GlobalStyleMemory update.
      - Capture the source text and component scores (r_entity, r_sem) of the
        best sample so MemoryUpdateCallback can write node history.
      - Pass current_step to reward functions via kwargs so history-conditioned
        weighting can compute the staleness factor correctly.
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Publish current step via module-level bus so reward functions can
        # compute stale_factor (inputs may be a list in newer TRL versions).
        common._current_step = self.state.global_step

        loss = super().training_step(model, inputs, num_items_in_batch)
        try:
            batch = inputs if isinstance(inputs, dict) else {}
            completions = batch.get("completions", [])
            answers = batch.get("answer", [])   # source texts
            rewards = self._metrics.get("rewards", [])
            component_scores = common._last_step_component_scores  # set by reward func

            if completions and len(rewards) > 0:
                reward_vals = [float(r) for r in rewards]
                best_idx = int(np.argmax(reward_vals))

                if best_idx < len(completions):
                    content = completions[best_idx]
                    if isinstance(content, list) and content:
                        best_text = content[0].get("content", "")
                    elif isinstance(content, dict):
                        best_text = content.get("content", "")
                    else:
                        best_text = str(content)

                    if best_text:
                        self.state._gsm_best_completion = best_text
                        self.state._gsm_best_reward = reward_vals[best_idx]

                        # Store source text for history update
                        if best_idx < len(answers):
                            self.state._gsm_best_source = answers[best_idx]

                        # Store component scores for the judger
                        if best_idx < len(component_scores):
                            cs = component_scores[best_idx]
                            # r_entity back-computed: privacy = 1 + r_entity
                            self.state._gsm_best_scores = {
                                "r_entity": cs["privacy"] - 1.0,
                                "r_sem": cs["utility"],
                            }
        except Exception:
            pass  # non-critical; training continues without memory update
        return loss


# ── Utilities ─────────────────────────────────────────────────────────────────


def random_sample_sentences(sentences, sample_size=4000, random_state=42):
    if len(sentences) <= sample_size:
        print(
            f"Dataset size ({len(sentences)}) ≤ sample size ({sample_size}). Using all data."
        )
        return sentences
    random.seed(random_state)
    sampled = random.sample(sentences, sample_size)
    print(f"Sampled {len(sampled)} sentences from {len(sentences)} total.")
    return sampled


# ── Main training loop ────────────────────────────────────────────────────────


def main():
    if DATASET_NAME not in DATASET_PATH:
        raise ValueError(f"Dataset '{DATASET_NAME}' not found in DATASET_PATH")

    csv_data = pd.read_csv(DATASET_PATH[DATASET_NAME])
    if TEXT_FIELD not in csv_data.columns:
        raise ValueError(f"Dataset must contain '{TEXT_FIELD}' column")

    sentences = csv_data[TEXT_FIELD].dropna().tolist()
    print(f"Total training data size: {len(sentences)}")

    sentences = random_sample_sentences(sentences, sample_size=4000)
    random.shuffle(sentences)

    # ── Build GlobalStyleMemory ────────────────────────────────────────────────
    print("Building GlobalStyleMemory...")
    style_memory = GlobalStyleMemory(lambda_sensitivity=GSM_LAMBDA, merge_radius=GSM_MERGE_RADIUS)
    style_memory.fit(sentences)
    print(
        f"GlobalStyleMemory built: {len(sentences)} texts → "
        f"{len(style_memory._embeddings)} clusters (δ={GSM_MERGE_RADIUS}). "
        f"Outlier threshold τ = {style_memory.tau:.4f}"
    )

    # Expose to rewards.py via common module
    common.style_memory = style_memory

    # ── Build dataset with memory-aware prompts ───────────────────────────────
    sentence_with_field = [{"review": remove_empty_rows(s)} for s in sentences]
    target_dataset = get_dataset(
        DATASET_NAME, sentence_with_field, style_memory=style_memory
    )

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        # 0.55 leaves ~3.4 GB for GRPO backward pass on a 8 GB GPU.
        # Raise to 0.7 only if running on ≥16 GB VRAM.
        gpu_memory_utilization=0.75,
    )

    # Paper §4.3: RL stage uses all attention + MLP projection layers
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    # ── GRPO training config ──────────────────────────────────────────────────
    training_args = GRPOConfig(
        learning_rate=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # 2 is the GRPO minimum and halves generation + logit memory vs 4.
        num_generations=2,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=EXPERIMENT_ID,
    )

    # ── LLM judger for history scoring ────────────────────────────────────────
    # Proxy mode by default (LLM_JUDGE_ENABLED=False in config): zero overhead.
    # Set LLM_JUDGE_ENABLED=True to use an LLM for explicit scoring.
    judger = LLMJudger()

    # ── Train ─────────────────────────────────────────────────────────────────
    import os
    log_path = os.path.join(EXPERIMENT_ID, "reward_components.jsonl")
    os.makedirs(EXPERIMENT_ID, exist_ok=True)

    if ABLATION_ENTROPY_ONLY:
        print("=" * 60)
        print("ABLATION MODE: entropy-only reward (r_gsm / r_entity / r_sem zeroed)")
        print(f"Log → {log_path}")
        print("=" * 60)

    trainer = SelfEvolvingGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[adaptive_combined_reward_func],
        args=training_args,
        train_dataset=target_dataset,
        callbacks=[
            MemoryUpdateCallback(
                style_memory,
                judger=judger,
                max_nodes=GSM_MAX_NODES,
                min_reward=GSM_MIN_REWARD_TO_UPDATE,
            ),
            RewardComponentLogger(log_path),
        ],
    )
    trainer.train()

    model_version = f"final_lora_{DATASET_NAME}_{EXPERIMENT_ID}"
    model.save_pretrained(model_version)
    tokenizer.save_pretrained(model_version)
    print(f"Model saved to {model_version}")


if __name__ == "__main__":
    main()
