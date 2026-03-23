import re
import spacy
from bert_score import BERTScorer
from utils import GlobalStyleMemory
from author import load_pretrained_author_classifier, author_id_field


# ── NLP pipeline ──────────────────────────────────────────────────────────────
nlp_model = spacy.load("en_core_web_sm")

# ── Semantic scorer ───────────────────────────────────────────────────────────
bert_scorer = BERTScorer(model_type="roberta-large", lang="en")

# ── Global Style Memory ───────────────────────────────────────────────────────
# Populated at runtime by train.py after GlobalStyleMemory.fit(sentences).
# rewards.reward_gsm_func() reads this via `from common import style_memory`.
style_memory: GlobalStyleMemory | None = None

# ── Per-step component score bus ──────────────────────────────────────────────
# adaptive_combined_reward_func writes individual component scores here after
# each reward call.
#
# Each entry is a dict with keys:
#   "privacy"  : float  — proxy privacy score ∈ [0,1]  (for MemoryUpdateCallback)
#   "utility"  : float  — proxy utility score ∈ [0,1]  (for MemoryUpdateCallback)
#   "r_entity" : float  — entity/nominal privacy reward
#   "r_gsm"    : float  — GlobalStyleMemory style reward
#   "r_entropy": float  — author-classification entropy reward
#   "r_sem"    : float  — BERTScore semantic utility reward
#   "r_infer"  : float  — attribute-inference penalty (0 when disabled)
#   "r_total"  : float  — final weighted combined reward
#   "is_outlier": bool  — whether source text was a GSM outlier
# List of dicts: one entry per sample in the current batch.
_last_step_component_scores: list = []

# ── Current training step bus ──────────────────────────────────────────────────
# Set by SelfEvolvingGRPOTrainer.training_step before calling super().
# Read by adaptive_combined_reward_func for staleness-decay computation.
_current_step: int = 0

# ── Author classification model ───────────────────────────────────────────────
author_model, author_tokenizer = load_pretrained_author_classifier(author_id_field)
if author_model is not None:
    author_model.eval()
else:
    print("Warning: Author model not loaded. author_classification_entropy_func will return 0.")


# ── Text utilities ─────────────────────────────────────────────────────────────
def extract_entities(text) -> set:
    doc = nlp_model(text)
    return set([ent.text for ent in doc.ents])


def extract_nominals(text) -> set:
    doc = nlp_model(text)
    subjects = [tok.text for tok in doc if tok.dep_ in ("nsubj", "nsubjpass")]
    objects = [tok.text for tok in doc if tok.dep_ in ("dobj", "pobj", "obj")]
    proper_nouns = [tok.text for tok in doc if tok.pos_ == "PROPN"]
    pronouns = [tok.text for tok in doc if tok.pos_ == "PRON"]
    return set(subjects + objects + proper_nouns + pronouns)


def mask_entities_and_contacts(text):
    doc = nlp_model(text)
    masked_text = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            start, end = ent.start_char, ent.end_char
            masked_text = masked_text[:start] + "*****" + masked_text[end:]
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    masked_text = re.sub(email_regex, "*****", masked_text)
    ssn_regex = r"\b\d{3}-\d{2}-\d{4}\b"
    masked_text = re.sub(ssn_regex, "*****", masked_text)
    phone_regex = r"\b\d{3}-\d{3}-\d{4}\b"
    masked_text = re.sub(phone_regex, "*****", masked_text)
    return masked_text
