from datasets import Dataset
from prompt_template import get_prompt
from config import USE_REASONING
from common import extract_entities

DATASET_PATH = {
    "yelp": "clean_data/yelp_train_split.csv",
    "tweet": "clean_data/filtered_twitter_data_5000.csv",
    "imdb": "clean_data/imdb_train_split.csv",
    "synpai": "clean_data/SynthPAI_AuthorInfo_train.csv",
    "all": "clean_data/filtered_yelp_data_5000.csv",
}


def rewrite_prompt_template(
    example: str,
    ref_text: str = None,
    is_outlier: bool = False,
    pii_list: list = None,
) -> str:
    """
    Build the user-turn rewriting prompt.

    Args:
        example:    Source text to rewrite.
        ref_text:   Style reference snippet sampled from GlobalStyleMemory.
                    When provided, the model is instructed to match this style.
        is_outlier: When True, adds an explicit guidance note instructing the
                    model to adopt neutral, formal tone (paper §3.3.1).
        pii_list:   Pre-computed PII entity list; extracted from example if None.
    """
    entities = pii_list if pii_list else list(extract_entities(example))
    pii_infos = ", ".join(entities) if entities else "none detected"

    outlier_hint = (
        "\n[NOTE: This text exhibits distinctive stylistic markers. "
        "Please adopt a neutral, formal tone to reduce re-identification risk.]\n"
        if is_outlier else "\n"
    )

    if ref_text is not None:
        return (
            f"Please paraphrase the following Input text. "
            f"Replace these PII: [{pii_infos}].{outlier_hint}"
            f"Adopt the style of this reference: \"{ref_text[:150]}\"\n"
            f"Input: {example}\nOutput:"
        )
    else:
        return (
            f"Please paraphrase the following Input text. "
            f"Replace these PII: [{pii_infos}].{outlier_hint}"
            f"Input: {example}\nOutput:"
        )


def _make_prompt_fn(style_memory, SYSTEM_PROMPT):
    """
    Returns a HuggingFace dataset map function that builds the full prompt
    for a single example, optionally using style_memory for reference sampling.
    """
    def make_prompt(x):
        text = x["review"]
        ref_text, is_out = None, False

        if style_memory is not None:
            is_out, _ = style_memory.is_outlier(text)
            ref_text = (
                style_memory.sample_reference_common()
                if is_out
                else style_memory.sample_reference_diverse(text)
            )

        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": rewrite_prompt_template(
                    text, ref_text=ref_text, is_outlier=is_out
                )},
            ],
            "answer": text,
        }
    return make_prompt


def get_yelp_data(data_list, style_memory=None):
    data = Dataset.from_list(data_list)
    SYSTEM_PROMPT = get_prompt()
    return data.map(
        _make_prompt_fn(style_memory, SYSTEM_PROMPT),
        remove_columns=data.column_names,
    )


def get_tweet_data(data_list, style_memory=None):
    data = Dataset.from_list(data_list)
    SYSTEM_PROMPT = get_prompt()
    return data.map(
        _make_prompt_fn(style_memory, SYSTEM_PROMPT),
        remove_columns=data.column_names,
    )


def get_imdb_data(data_list, style_memory=None):
    data = Dataset.from_list(data_list)
    SYSTEM_PROMPT = get_prompt()
    return data.map(
        _make_prompt_fn(style_memory, SYSTEM_PROMPT),
        remove_columns=data.column_names,
    )


def get_synpai_data(data_list, style_memory=None):
    data = Dataset.from_list(data_list)
    SYSTEM_PROMPT = get_prompt()
    return data.map(
        _make_prompt_fn(style_memory, SYSTEM_PROMPT),
        remove_columns=data.column_names,
    )


def get_sft_data(data_list, tokenizer):
    """
    Build a supervised fine-tuning dataset from pseudo-parallel pairs.
    Each item in data_list must have: 'review', 'answer'.
    Optionally: 'ref_text' (style reference) and 'is_outlier' (bool).
    The style reference is included in the prompt so the model learns
    to condition on it during SFT warmup (paper §3.2).
    """
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    SYSTEM_PROMPT = get_prompt()
    data_sft = []
    for d in data_list:
        data_sft.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": rewrite_prompt_template(
                        d["review"],
                        ref_text=d.get("ref_text"),
                        is_outlier=bool(d.get("is_outlier", False)),
                    ),
                },
                {"role": "assistant", "content": d["answer"]},
            ]
        })

    dataset_sft = Dataset.from_list(data_sft)
    dataset_sft = dataset_sft.map(formatting_prompts_func, batched=True)
    return dataset_sft


def remove_empty_rows(text: str) -> str:
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    return "\n".join(non_empty_lines)


def get_dataset(dataset_name, aug_sentences, style_memory=None):
    """
    Route to the correct dataset loader.
    style_memory is passed through to enable adaptive prompt construction.
    """
    if dataset_name in ("yelp", "all"):
        return get_yelp_data(aug_sentences, style_memory=style_memory)
    elif dataset_name == "tweet":
        return get_tweet_data(aug_sentences, style_memory=style_memory)
    elif dataset_name == "imdb":
        return get_imdb_data(aug_sentences, style_memory=style_memory)
    elif dataset_name == "synpai":
        return get_synpai_data(aug_sentences, style_memory=style_memory)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
