import argparse
import json
import logging
import random
import shelve
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from legal_testing.datasets.ECHR19.utils import (
    DATASET_PATH,
    load_echr19_subset,
    store_echr19_subset,
)
from legal_testing.util.gpt_interface import TokenCounter, gpt_query

FactIndex = int
Fact = str

LLM_SYSTEM_PROMPT = """You are a superintelligent expert knowledge system which takes user queries and answers them as precisely and truthfully as possible while coming up with creative ideas and solutions."""  # noqa

LLM_PREFIX = """### Description
Create multiple paraphrases of  the following legal fact. The paraphrased fact must contain the exact same information as the original fact and must be written in the same formal style. Use the following output format:

[START]
1]] "First paraphrasing of original legal fact"

2]] "Second paraphrasing of original legal fact"

3]] "Third paraphrasing of original legal fact"
[END]

The first line must only contain the [START] token and the last line must only contain the [END] token.

### Original legal fact
"""  # noqa

LLM_SUFFIX = """

### Paraphrases
[START]
"""


DEFAULT_MODEL_PARAMS = {
    "model_name": "gpt-3.5-turbo",
    "max_tokens": 2048,
    "temperature": 0.8,
}


def _build_prompt(
    original_fact: str,
) -> str:
    """Given an original fact, build a prompt which can be used to generate paraphrases.

    Args:
        original_fact (str): The original fact.

    Returns:
        str: The prompt.
    """
    return LLM_PREFIX + original_fact + LLM_SUFFIX


def _extract_paraphrased_facts(
    query_completion: str,
) -> List[Fact]:
    """Extract the paraphrased facts from the query completion provided by the LLM.

    Args:
        query_completion (str): The query completion provided by the LLM.

    Raises:
        ValueError: If the query completion is invalid.

    Returns:
        List[Fact]: The extracted paraphrased facts.
    """
    paraphrases: List[Fact] = []

    # Check that the query completion is valid
    if not (query_completion.startswith("1]]") and query_completion.endswith("[END]")):
        raise ValueError("Invalid query completion")

    # Compute the number of paraphrases
    num_paraphrases = query_completion.count("]]")

    # Remove the [END] token and append a fake list item at the very end
    # to make subsequent processing easier
    query_completion = query_completion.replace("[END]", f"{num_paraphrases+1}]]")

    # Extract all paraphrases
    for i in range(1, num_paraphrases + 1):
        # Extract the paraphrase
        paraphrase = query_completion.split(f"{i}]]")[1].split(f"{i+1}]]")[0]
        paraphrase = paraphrase.strip()

        # Add the paraphrase to the list
        paraphrases.append(paraphrase)

    return paraphrases


def _generate_paraphrases(
    fact_paraphraser: Callable[[str], str],
    facts: List[str],
    num_attacks_per_case: int = 1,
    filter_bad_facts: bool = True,
    model_params: Optional[Dict[str, Any]] = None,
) -> List[Tuple[FactIndex, Fact]]:
    """Given a list of facts, select 'num_attacks_per_case' facts to attack and generate
    paraphrases for them.

    Args:
        fact_paraphraser (Callable[[str], str]): The fact paraphraser to use.
        facts (List[str]): The list of facts from which to select the facts to attack.
        num_attacks_per_case (int, optional): The number of facts to select for attack.
            Defaults to 1.
        filter_bad_facts (bool, optional): Whether to filter out facts which are too short.
        model_params (Optional[Dict[str, Any]], optional): The parameters to pass to the fact
            paraphraser. Defaults to None.

    Returns:
        List[Tuple[FactIndex, Fact]]: The list of facts to attack together with their indices.
    """
    # Create a copy of the facts
    facts = facts.copy()
    fact_tuples = [(index, fact) for index, fact in enumerate(facts)]

    facts_to_attack_tuples: List[Tuple[FactIndex, Fact]] = []

    if filter_bad_facts:
        # Filter out facts which are shorter than 120 characters
        fact_tuples = [tup for tup in fact_tuples if len(tup[1]) > 119]

        # Filter out the first fact
        fact_tuples = fact_tuples[1:]

    # Select the facts to attack
    # We have 'num_attacks_per_case' attacks per case
    replace = len(fact_tuples) < num_attacks_per_case
    if replace:
        selected_tuples = random.choices(fact_tuples, k=num_attacks_per_case)
    else:
        selected_tuples = random.sample(fact_tuples, k=num_attacks_per_case)

    facts_to_attack_tuples.extend(selected_tuples)

    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS

    # Create the paraphrased facts
    paraphrased_fact_tuples: List[Tuple[FactIndex, Fact]] = []

    for fact_index, fact_to_attack in facts_to_attack_tuples:
        # First remove the fact number from the beginning of the fact
        splitted_fact = fact_to_attack.split(" ", 1)
        if splitted_fact[0][0].isdigit() and len(splitted_fact) > 1:
            fact_number_prefix, fact_to_attack = splitted_fact[0] + " ", splitted_fact[1]
        else:
            fact_number_prefix = ""

        # Build the prompt
        prompt = _build_prompt(
            original_fact=fact_to_attack,
        )

        # Generate the paraphrased facts
        query_completion = fact_paraphraser(
            prompt,
            **model_params,
        )

        # Extract the paraphrased facts
        try:
            # This is purely for debugging purposes
            if query_completion.startswith("[DEBUG]"):
                paraphrased_facts = [query_completion]
            else:
                paraphrased_facts = _extract_paraphrased_facts(
                    query_completion=query_completion,
                )
        except (ValueError, IndexError) as e:
            # If parsing failed, append an error message together with the faulty query completion
            paraphrased_fact_tuples.append((fact_index, "[ERROR]" + query_completion + str(e)))
        else:
            # Append the paraphrased facts to the result list
            paraphrased_fact_tuples.extend(
                (fact_index, fact_number_prefix + fact) for fact in paraphrased_facts
            )

    return paraphrased_fact_tuples


def create_paraphrased_facts(
    samples: List[Dict[str, Any]],
    fact_paraphraser: Callable,
    cache_file_path: str,
    model_params: Dict[str, Any] = None,
    num_attacks_per_case: int = 1,
    attack_most_important_fact: bool = False,
) -> List[Dict[str, Any]]:
    """Given a list of legal cases, create paraphrases for the facts in the cases.

    Args:
        samples (List[Dict[str, Any]]): The list of legal cases.
        fact_paraphraser (Callable): The fact paraphraser to use.
        cache_file_path (str): The path to the cache file.
        model_params (Dict[str, Any], optional): The parameters to pass to the fact paraphraser.
        num_attacks_per_case (int, optional): The number of facts to attack per case.
        attack_most_important_fact (bool, optional): Whether to additionally attack the most
            important fact.

    Returns:
        List[Dict[str, Any]]: The list of legal cases extended by a list of paraphrased facts
            for each case.
    """

    # Create the cache directory if it does not exist
    Path(cache_file_path).parent.mkdir(parents=True, exist_ok=True)

    with shelve.open(cache_file_path) as cache:
        for index, sample in enumerate(samples):
            # Extract the case ID
            case_id = sample["ITEMID"]

            logging.info(f"Generating paraphrases for case {index + 1}/{len(samples)}: {case_id}")

            # Check if the case ID is already in the cache
            if case_id in cache:
                sample["PARAPHRASES_RANDOM"] = cache[case_id]["PARAPHRASES_RANDOM"]

                if attack_most_important_fact:
                    sample["PARAPHRASES_MOST_IMPORTANT_FACT"] = cache[case_id][
                        "PARAPHRASES_MOST_IMPORTANT_FACT"
                    ]
                continue

            # Extract the original fact
            original_facts = sample["TEXT"]

            data_for_cache: Dict[str, List[Tuple[FactIndex, Fact]]] = {}

            # Attack the most important fact if specified
            if attack_most_important_fact:
                attention_weights: List[float] = sample["ATTENTION_WEIGHTS"]

                # Get the most important fact
                most_important_fact_index = attention_weights.index(max(attention_weights))
                most_important_fact = original_facts[most_important_fact_index]

                # Attack the most important fact
                paraphrased_fact_tuples: List[Tuple[FactIndex, Fact]] = _generate_paraphrases(
                    fact_paraphraser=fact_paraphraser,
                    facts=[most_important_fact],
                    num_attacks_per_case=1,
                    filter_bad_facts=False,
                )

                # Replace the fact indices of the paraphrased facts with the most
                # important fact index
                paraphrased_fact_tuples = [
                    (most_important_fact_index, fact) for (_, fact) in paraphrased_fact_tuples
                ]

                logging.info("Created the following paraphrases for the most important fact:")
                for _, fact in paraphrased_fact_tuples:
                    logging.info(f"\t- {fact}")

                sample["PARAPHRASES_MOST_IMPORTANT_FACT"] = paraphrased_fact_tuples
                data_for_cache["PARAPHRASES_MOST_IMPORTANT_FACT"] = paraphrased_fact_tuples

            # Attack random facts
            paraphrased_fact_tuples = _generate_paraphrases(
                fact_paraphraser=fact_paraphraser,
                facts=original_facts,
                num_attacks_per_case=num_attacks_per_case,
                model_params=model_params,
                filter_bad_facts=True,
            )

            logging.info("Created the following paraphrases for random facts:")
            for _, fact in paraphrased_fact_tuples:
                logging.info(f"\t- {fact}")

            data_for_cache["PARAPHRASES_RANDOM"] = paraphrased_fact_tuples

            # Add the paraphrased facts to the cache
            cache[case_id] = data_for_cache

            # Add the paraphrased facts to the sample
            sample["PARAPHRASES_RANDOM"] = paraphrased_fact_tuples

    return samples


def fix_error(fact: str) -> List[str]:
    """Fix failed parsings of the LLM and try to extract the paraphrases.

    Args:
        fact (str): The fact to fix.

    Raises:
        NotImplementedError: If the error type is not supported.

    Returns:
        List[str]: The fixed paraphrases.
    """
    # Remove the error prefix
    fact = fact.replace("[ERROR]", "")

    # Check which type of error occurred
    if fact.startswith("1]]"):
        if "[END]" in fact:
            # In this case there was some issue with extracting the paraphrase numbers (e.g. 1]])
            # Remove everything after the [END] token
            fact = fact[: fact.index("[END]") + len("[END]")]
            discard_last_paraphrase = False
        else:
            # In this case the fact is missing an [END] token and the last paraphrase might not
            # be complete.
            # Add it to the end of the fact
            fact += "\n[END]"
            discard_last_paraphrase = True

        # Extract the facts
        paraphrases = _extract_paraphrased_facts(fact)

        if discard_last_paraphrase:
            # The last paraphrase might be broken due to the missing [END] token. Remove it.
            paraphrases = paraphrases[:-1]

        return paraphrases

    else:
        raise NotImplementedError


def fix_errors(subset_path: str) -> None:
    """Fix failed parsings of the LLM and try to extract the paraphrases.

    Args:
        subset_path (str): The path to the data subset to fix.
    """
    # Load the samples
    with open(subset_path, "r") as f:
        samples = json.load(f)

    # Iterate over all samples
    for sample in samples:
        # Perform the error fixing for all attack types
        for attack_type in ["PARAPHRASES_RANDOM", "PARAPHRASES_MOST_IMPORTANT_FACT"]:
            if attack_type not in sample:
                continue

            fixed_paraphrases = []
            paraphrases_to_remove = []
            # Iterate over all attacks of the current type
            for paraphrase_index, (fact_index, fact) in enumerate(sample[attack_type]):
                # check for errors
                if "ERROR" in fact:
                    # Fix the error
                    fixed_sentences = fix_error(fact)

                    # Extract the fact_number from the original fact
                    original_fact = sample["TEXT"][fact_index]
                    splitted_fact = original_fact.split(" ")
                    if splitted_fact[0][0].isdigit() and len(splitted_fact) > 1:
                        fact_number_prefix = splitted_fact[0] + " "
                    else:
                        fact_number_prefix = ""

                    # Build the fixed paraphrases
                    fixed_paraphrases.extend(
                        (fact_index, fact_number_prefix + fixed_sentence)
                        for fixed_sentence in fixed_sentences
                    )
                    paraphrases_to_remove.append(paraphrase_index)

            sample[attack_type].extend(fixed_paraphrases)

            # Remove the paraphrases that were broken
            for index in sorted(paraphrases_to_remove, reverse=True):
                del sample[attack_type][index]

    # Save the samples
    subset_path = Path(subset_path)
    subset_path = str(subset_path.with_suffix("")) + "_fixed.json"

    with open(subset_path, "w") as f:
        json.dump(samples, f, indent=4)


def clean_paraphrases(subset_path: str) -> None:
    """Clean the paraphrases from failed parsings of the LLM.

    Args:
        subset_path (str): The path to the data subset to clean.
    """
    # Load the samples
    with open(subset_path, "r") as f:
        samples = json.load(f)

    # Iterate over all samples
    for sample in samples:
        for attack_type in ["PARAPHRASES_RANDOM", "PARAPHRASES_MOST_IMPORTANT_FACT"]:
            if attack_type not in sample:
                continue

            # Remove all excessive points, as well as excessive whitespace
            new_samples = []
            for fact_index, fact in sample[attack_type]:
                if "]]" in fact:
                    # Check if there is a digit before the ]] token
                    if fact.split("]]", 1)[0][-1].isdigit():
                        # Only keep text before the first ]] token
                        fact = fact.split("]]", 1)[0][:-1]
                    print(f"Excessive ]] token found! in case {sample['ITEMID']}")
                new_samples.append((fact_index, fact.strip()))

            sample[attack_type] = new_samples

    # Save the samples
    subset_path = Path(subset_path)
    subset_path = str(subset_path.with_suffix("")) + "_cleaned.json"

    with open(subset_path, "w") as f:
        json.dump(samples, f, indent=4)


def main(
    subset_type: str,
    subset_size: int,
    name_suffix: Optional[str] = None,
    num_attacks_per_case: int = 1,
    attack_most_important_fact: bool = False,
):
    """Create paraphrases for the legal cases contained in the provided data subset.

    Args:
        subset_type (str): The type of subset to load.
        subset_size (int): The size of the subset which should be loaded.
        name_suffix (Optional[str], optional): A custom suffix which is appended to the
            subset_name before loading. Defaults to None.
        num_attacks_per_case (int, optional): The number of attacks executed for each legal case.
        attack_most_important_fact (bool, optional): Whether additionally the most important fact
            should be attacked.
    """
    # Load the test set samples
    samples = load_echr19_subset(
        subset_type=subset_type,
        subset_size=subset_size,
        name_suffix=name_suffix,
        attention_weights=attack_most_important_fact,
    )

    # Prepare the fact paraphraser
    fact_paraphraser = gpt_query
    # fact_paraphraser = TokenCounter(max_tokens=2048)

    # Create a cache file path
    cache_file_path = (
        DATASET_PATH
        + f"/cache_subset_{subset_type}_size_{subset_size}_num_attacks_{num_attacks_per_case}"
        f"_attack_most_important_{attack_most_important_fact}.db"
    )

    # Create the additional facts
    extended_samples = create_paraphrased_facts(
        samples=samples,
        fact_paraphraser=fact_paraphraser,
        cache_file_path=cache_file_path,
        num_attacks_per_case=num_attacks_per_case,
        attack_most_important_fact=attack_most_important_fact,
    )

    # Print the number of tokens
    # logging.info(f"Number of input tokens: {fact_paraphraser.input_token_counter}")
    # logging.info(f"Number of output tokens: {fact_paraphraser.output_token_counter}")
    # logging.info(f"Total number of tokens: {fact_paraphraser.total_token_counter}")
    # logging.info(f"Estimated cost upper bound: {fact_paraphraser.cost_upper_bound:.2f}")
    # logging.info(
    #     f"10 largest input token sizes: {fact_paraphraser.top_k_input_token_lengths(k=10)}"
    # )

    # Store the samples
    if name_suffix is None:
        name_suffix = ""
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name_suffix += (
        f"_extended_{num_attacks_per_case + int(attack_most_important_fact)}"
        f"_attacks_per_case_{current_date_time}"
    )
    store_echr19_subset(
        subset=extended_samples,
        subset_type=subset_type,
        name_suffix=name_suffix,
        attention_weights=attack_most_important_fact,
    )


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Which experiment should be run.", required=True, choices=["create_paraphrases", "fix_errors", "clean_paraphrases"], default="create_paraphrases") # noqa
    parser.add_argument("--subset_type", type=str, help="[CREATE_PARAPHRASES] The type of subset to load", default="logit_equal_distribution") # noqa
    parser.add_argument("--subset_size", type=int, help="[CREATE_PARAPHRASES] The size of the subset which should be loaded", default=500) # noqa
    parser.add_argument("--name_suffix", type=str, help="[CREATE_PARAPHRASES] A custom suffix which is appended to the subset_name before loading", default="") # noqa
    parser.add_argument("--num_attacks_per_case", type=int, help="[CREATE_PARAPHRASES] The number of attacks executed for each legal case", default=10) # noqa
    parser.add_argument("--attack_most_important_fact", action="store_true", help="[CREATE_PARAPHRASES] Whether additionally the most important fact should be attacked.", default=True) # noqa
    parser.add_argument("--subset_path", type=str, help="[FIX_ERRORS / CLEAN_PARAPHRASES] An path to a given subset", default="") # noqa
    # fmt: on
    args = parser.parse_args()

    # Set up the logger
    logging.basicConfig(
        format="▸ %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    if args.mode == "create_paraphrases":
        main(
            subset_type=args.subset_type,
            subset_size=args.subset_size,
            name_suffix=args.name_suffix,
            num_attacks_per_case=args.num_attacks_per_case,
            attack_most_important_fact=args.attack_most_important_fact,
        )
    else:
        assert args.subset_path != "", "You must provide the path to a subset to process!"

        if args.mode == "fix_errors":
            fix_errors(args.subset_path)

        elif args.mode == "clean_paraphrases":
            clean_paraphrases(args.subset_path)
