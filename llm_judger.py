"""
llm_judger.py — Privacy and utility scoring for GlobalStyleMemory history.

Provides two operating modes, controlled by LLM_JUDGE_ENABLED in config.py:

  Proxy mode (default, LLM_JUDGE_ENABLED=False):
    Derives (privacy, utility) from precomputed reward values that are already
    available at training time. Zero additional GPU overhead.
        privacy = clip(1 + r_entity, 0, 1)   [r_entity ∈ [-1,0] → privacy ∈ [0,1]]
        utility = clip(r_sem, 0, 1)           [r_sem already ∈ [0,1]]

  LLM judge mode (LLM_JUDGE_ENABLED=True):
    Calls a frozen judge LLM with a structured scoring prompt and parses its
    response into (privacy ∈ [0,1], utility ∈ [0,1]). Falls back to proxy
    scores if parsing fails. Adds ~200 ms per step; recommended only on
    machines with spare VRAM or when running the judger on a second GPU.

In both modes the output format is identical, so the rest of the codebase
(GlobalStyleMemory.update_node_history, MemoryUpdateCallback) is oblivious
to which mode is active.
"""

import re
import numpy as np

from config import LLM_JUDGE_ENABLED, LLM_JUDGE_MODEL


class LLMJudger:
    """
    Score (source, rewrite) pairs for privacy and utility.

    Usage
    -----
    judger = LLMJudger()

    # Proxy mode (no LLM call):
    privacy, utility = judger.score_from_rewards(r_entity=-0.2, r_sem=0.85)

    # LLM mode (requires LLM_JUDGE_ENABLED=True):
    privacy, utility = judger.score_with_llm(source="...", rewrite="...")
    """

    # Structured prompt: forces the model to emit two labelled floats.
    _JUDGE_PROMPT = (
        "You are a privacy evaluator. Assess the following rewrite.\n\n"
        "Original: {source}\n"
        "Rewrite:  {rewrite}\n\n"
        "Rate on two dimensions (0.0–1.0):\n"
        "  PRIVACY: 1.0 = no identifying information remains; "
        "0.0 = original author fully identifiable.\n"
        "  UTILITY: 1.0 = identical meaning preserved; "
        "0.0 = meaning completely lost.\n\n"
        "Reply with EXACTLY two lines:\n"
        "PRIVACY: <score>\n"
        "UTILITY: <score>"
    )
    _SCORE_RE = re.compile(
        r"PRIVACY:\s*([\d.]+).*?UTILITY:\s*([\d.]+)", re.DOTALL | re.IGNORECASE
    )

    def __init__(self):
        self._model = None
        self._tokenizer = None
        if LLM_JUDGE_ENABLED:
            self._load_judge()

    def _load_judge(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[LLMJudger] Loading judge model: {LLM_JUDGE_MODEL}")
        self._tokenizer = AutoTokenizer.from_pretrained(LLM_JUDGE_MODEL)
        self._model = AutoModelForCausalLM.from_pretrained(
            LLM_JUDGE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._model.eval()
        print("[LLMJudger] Judge model loaded.")

    # ── Proxy mode ─────────────────────────────────────────────────────────────

    def score_from_rewards(
        self, r_entity: float, r_sem: float
    ) -> tuple:
        """
        Derive (privacy, utility) from already-computed reward values.

        r_entity ∈ [-1, 0]:   0 = no entities leaked (privacy=1)
                              -1 = all entities leaked (privacy=0)
        r_sem    ∈ [0, 1]:    maps directly to utility score.

        No LLM call; uses reward values from the current training step.
        """
        privacy = float(np.clip(1.0 + r_entity, 0.0, 1.0))
        utility = float(np.clip(r_sem, 0.0, 1.0))
        return privacy, utility

    # ── LLM judge mode ─────────────────────────────────────────────────────────

    def score_with_llm(self, source: str, rewrite: str) -> tuple:
        """
        Call the judge LLM to score (source, rewrite).

        Requires LLM_JUDGE_ENABLED=True in config.py. Falls back to neutral
        scores (0.5, 0.5) if the model output cannot be parsed.

        Returns (privacy ∈ [0,1], utility ∈ [0,1]).
        """
        if self._model is None:
            raise RuntimeError(
                "LLM judger not loaded. Set LLM_JUDGE_ENABLED=True in config.py."
            )

        import torch

        prompt = self._JUDGE_PROMPT.format(
            source=source[:400], rewrite=rewrite[:400]
        )
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the generated portion
        input_len = inputs["input_ids"].shape[1]
        response = self._tokenizer.decode(
            out[0, input_len:], skip_special_tokens=True
        )

        m = self._SCORE_RE.search(response)
        if m:
            privacy = float(np.clip(float(m.group(1)), 0.0, 1.0))
            utility = float(np.clip(float(m.group(2)), 0.0, 1.0))
        else:
            print(
                f"[LLMJudger] Warning: could not parse judge response: {response!r}. "
                "Falling back to neutral scores (0.5, 0.5)."
            )
            privacy, utility = 0.5, 0.5

        return privacy, utility

    # ── Unified entry point ─────────────────────────────────────────────────────

    def score(
        self,
        source: str,
        rewrite: str,
        r_entity: float = 0.0,
        r_sem: float = 0.5,
    ) -> tuple:
        """
        Return (privacy, utility) using whichever mode is active.

        If LLM_JUDGE_ENABLED is True, calls score_with_llm and ignores
        the proxy values. Otherwise uses score_from_rewards.
        """
        if LLM_JUDGE_ENABLED:
            return self.score_with_llm(source, rewrite)
        return self.score_from_rewards(r_entity, r_sem)
