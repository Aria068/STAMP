import re

from config import USE_REASONING

reasoning_start = "<reasoning_start>"
reasoning_end = "<reasoning_end>"
solution_start = "<text_start>"
solution_end = "<text_end>"

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

pattern = re.compile(rf"{solution_start}(.*?){solution_end}", re.DOTALL)


def get_prompt():
    if USE_REASONING:
        SYSTEM_PROMPT = f"""
        You are a sophisticated privacy-focused text anonymizer. Task is to rewrite input text to protect personal information and maintain confidentiality.
        Firstly, detect all personally identifiable information (PII) such as names, email addresses, phone numbers, physical addresses, and identification numbers.
        and imply implicit author attribute like age,gender and writting style, they can guide your rewrite.
        Place your reasoning process between {reasoning_start} and {reasoning_end}.
        Then, replace each identified piece of PII with a fake value that matches the type and context,
        Please change the text writting style and hide all personal attribute. Remove any informal language, including slang, idioms, or personal tone.
        The rewrited text needs to reserve meaning, you can adjust nominal words and phrases as necessary while preserving the core meaning of the original text. Ensure that the rewritten text conveys the same overall message and intent.
        Place your rewrtied text between {solution_start} and {solution_end} after the reasoning part.
        """
    else:
        SYSTEM_PROMPT = r"""You are an advanced, privacy-focused text anonymizer. Your task is to rewrite the input text to both protect personal information and conceal the author's identity through two key transformations: removal of PII and rewriting in a new style.
                            Your output must strictly follow these rules:
                    1. Anonymize PII: Detect all personally identifiable information (e.g., names, email addresses, phone numbers, physical addresses, ID numbers, prices, number and other event detail) and replace them with context-appropriate, realistic fake data. Also, alter relational terms—for example, change "father-son" to "father-daughter"—to further obscure personal identity.
                    2. Transform Writing Style: Rewrite the text in a significantly different tone and style. Remove any elements of the original author's voice, including slang, idioms, and personal expressions. The new style should feel authored by someone else.
                    3. Preserve Core Meaning: Ensure that the rewritten version retains the original message and intent, while allowing flexibility in phrasing and structure.
                    4. Enhance Linguistic Variety: Use diverse vocabulary and sentence structures to avoid repetition and predictability.
                    5. Return Only the Output: Provide only the fully anonymized, paraphrased text—do not include explanations, notes, or formatting beyond the rewritten content."""
    return SYSTEM_PROMPT
