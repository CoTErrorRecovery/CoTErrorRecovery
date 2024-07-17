"""
Constants and common utilities for use across the code.
"""
import re
from pathlib import Path


# default tasksets for each dataset
DEFAULT_TASKSETS = {
    'awps': 'MultiArith',
    'asdiv': 'ASDiv',
    'svamp': 'SVAMP',
    'gsm8k': 'test'
}

# path to results directory
RESULTS_PATH = (Path(__file__).parent / 'results').abspath()

# file suffix to error type annotation
ERROR_TYPE = {
    'any': 'copying',
    'copy': 'copying',
    'calc': 'calculation',
    'propcalc': 'calculation'
}

# chain of thought prompt. From Kojima et al 2022 (https://arxiv.org/abs/2205.11916).
COT_PROMPT = " Let's think step by step."
RECOVERY_PROMPT = " Let's think step by step, being careful to notice and fix any mistakes."

# answer extraction strings. Also from Kojima et al 2022.
GSM8K_SUFFIX = " Therefore, the answer (arabic numerals) is"
DIRECT_SUFFIX = "The answer (arabic numerals) is"

# verbose output flag
VERBOSE = True


def generate_prompt(question: str, style: str = ""):
    """
    Generate a full prompt string, given a question.

    Args:
    - question: the question to be incorporated in the prompt

    Keyword Args:
    - style: the style of the prompt.
        - 'sbs' indicates zero-shot chain of thought prompt, and anything else indicates no prompt.

    Returns: the prompt text, incorporating the question
    """
    if "qa" in style or 'sbs' in style:
        if ('Q:' not in question) and ('Question:' not in question):
            question = 'Q: ' + question
        if ('\nA:' not in question) and ('\nAnswer:' not in question):
            question = question + '\n\nA:'
        question = question.replace("Q: Q:", "Q:")  # replace extra "Q:"
    if 'sbs' in style and COT_PROMPT not in question and RECOVERY_PROMPT not in question:  # zero-shot chain of thought
        if 'recovery' in style:  # error recovery CoT prompt
            question = question + RECOVERY_PROMPT
        else:
            question = question + COT_PROMPT
    if "zeroshot" in style:  # zero-shot direct
        question = question + ' ' + DIRECT_SUFFIX
    return question  # if style not identified, return question as-is

# answer scoring function


def number_scorer(generation: [str, int, float], target: [str, int, float]):
    """
    Check if the generation contains the target value.
    - Very similar to answer scoring from Kojima et al 2022 (https://arxiv.org/abs/2205.11916)

    Args:
    - generation: generated text to evaluate
    - target: the target value to search for

    Returns: whether the generation contains the target value (bool)
    """
    if not isinstance(generation, str):
        generation = str(generation)

    # remove the answer extraction string from the generation
    if GSM8K_SUFFIX in generation:
        generation = generation.split(GSM8K_SUFFIX)[-1]
    elif DIRECT_SUFFIX in generation:
        generation = generation.split(DIRECT_SUFFIX)[-1]
    generation = generation.replace(',', '')  # remove commas from number representations, e.g. $1,000 -> $1000
    number = re.search(r'-?\d+\.?\d*', generation)  # get first number (negative or positive, can include decimal)

    if not isinstance(target, (int, float)):
        target = eval(target)
    if number:
        return target == eval(str(number[0]))  # check if the identified number evaluates to the same value as the target.
    return False  # if no numerical match found, return False
