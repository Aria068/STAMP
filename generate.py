from unsloth import FastLanguageModel
import pickle as pkl
import spacy
from tqdm import tqdm
from vllm import SamplingParams
import argparse
import pandas as pd
from prompt_template import get_prompt
from config import lora_rank, max_seq_length
from dataset import (
    rewrite_prompt_template,
    remove_empty_rows,
)


class RLModel:
    def __init__(self, model_path=None, using_reasoning=False):
        if model_path is None:
            raise ValueError("model_path must be provided")

        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank  # Larger rank = smarter, but slower

        self.nlp_model = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
        self.SYSTEM_PROMPT = get_prompt()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,  # "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,  # False for LoRA 16bit
            fast_inference=True,  # Enable vLLM fast inference
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization=0.8,  # Reduce if out of memory
        )

        self._model = model
        self._tokenizer = tokenizer
        self._sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
        )

    def batch_generate(self, dataset, model_version, dataset_name):
        """Generate synthetic text for the entire dataset."""
        import os

        target_text_field = "review"
        original_texts, synthetic_texts = [], []

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        for d in tqdm(dataset):
            if target_text_field not in d:
                print(f"Warning: '{target_text_field}' field not found in data item")
                continue

            try:
                text = self._tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": rewrite_prompt_template(d[target_text_field]),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                output = (
                    self._model.fast_generate(
                        text,
                        sampling_params=self._sampling_params,
                    )[0]
                    .outputs[0]
                    .text
                )

                # Clean up output format
                if output.startswith("Here"):
                    syn_txt = output.split(":", 1)[-1].strip()
                else:
                    syn_txt = output

                original_texts.append(d[target_text_field])
                synthetic_texts.append(syn_txt)

            except Exception as e:
                print(f"Error processing item: {e}")
                continue

        # Save results
        try:
            df_syn = pd.DataFrame(synthetic_texts, columns=[target_text_field])
            csv_path = f"results/{model_version}_{dataset_name}_syn.csv"
            df_syn.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")

            pickle_path = f"results/{model_version}_{dataset_name}_syn.p"
            with open(pickle_path, "wb") as f:
                pkl.dump(
                    {
                        "origin": original_texts,
                        "synth": synthetic_texts,
                    },
                    f,
                )
            print(f"Pickle data saved to {pickle_path}")

        except Exception as e:
            print(f"Error saving results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL model generation.")
    parser.add_argument(
        "--model_version",
        type=str,
        default="final_lora_yelp_stamp_yelp",
        help="Path or name of the model version.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="tweet",
        help="Name of the dataset to use (e.g., tweet).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="text",
        help="Name of the dataset to use (e.g., tweet).",
    )

    args = parser.parse_args()
    dataset_path = args.dataset_path
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    text_col = args.text

    # Preprocess dataset
    csv_data = pd.read_csv(dataset_path)
    sentences = csv_data[text_col].tolist()

    sentence_with_field = []
    for i, s in enumerate(sentences):
        sentence_with_field.append(
            {
                "review": remove_empty_rows(s),
            }
        )

    model_version = args.model_version
    rl_model = RLModel(model_path=model_version)
    rl_model.batch_generate(
        dataset=sentence_with_field,
        model_version=model_version.split("/")[-1],
        dataset_name=dataset_name,
    )
