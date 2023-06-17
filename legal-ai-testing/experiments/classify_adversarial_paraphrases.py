import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict

from legal_testing.models.bert_echr_only_head import get_model, get_preprocessor

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).absolute().parent.parent / "data"
RESULT_DIR = Path(__file__).absolute().parent / "results"


def test_paraphrase_attacks(
    dataset_path: Path,
    model: Callable,
    preprocess: Callable,
):
    """Load the paraphrases from the dataset and use the model to classify the modified legal cases.

    Args:
        dataset_path (Path): The path to the dataset file storing the legal cases and paraphrases for
            selected facts.
        model (Callable): The model to use for classification.
        preprocess (Callable): The preprocessing function to use for the model.
    """
    # Load the dataset
    with open(dataset_path, "r") as file:
        samples = json.load(file)

    # Instantiate a result dictionary
    results: Dict[str, Any] = {}

    # Iterate over the samples
    for index, sample in enumerate(samples):
        logging.info(f"Attacking sample {index}/{len(samples)}")
        sample_id = sample["ITEMID"]

        results[sample_id] = {
            "original_logit": sample["LOGIT"],
            "original_prediction": sample["PREDICTION"],
            "results": [],
        }

        # Classify all paraphrased attacks
        for attack_type in ["PARAPHRASES_RANDOM", "PARAPHRASES_MOST_IMPORTANT_FACT"]:
            # Iterate over all paraphrases of this attack type
            for attack_index, paraphrase in sample[attack_type]:
                # Build the adversarial input
                adversarial_text = list(sample["TEXT"])
                adversarial_text[attack_index] = paraphrase
                adversarial_case = {
                    "TEXT": adversarial_text,
                    "VIOLATED_ARTICLES": sample["VIOLATED_ARTICLES"],
                }

                # Classify the adversarial input
                output = model(preprocess(adversarial_case))

                # Extract the result
                modified_logit = output.logits[0].item()
                modified_prediction = int(modified_logit > 0.5)

                # Store the result
                results[sample_id]["results"].append(
                    {
                        "attack_type": attack_type,
                        "attack_index": attack_index,
                        "paraphrase": paraphrase,
                        "logit": modified_logit,
                        "prediction": modified_prediction,
                    }
                )

    # Store the results
    dataset_path = str(Path(dataset_path).with_suffix("")) + "_results.json"

    with open(dataset_path, "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="The path to the dataset file storing the samples and their paraphrases.")
    # fmt: on

    args = parser.parse_args()

    # Set up the logger
    logging.basicConfig(
        format="▸ %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    # Prepare the classifier
    model = get_model(load_weights=True, return_attention_weights=True)

    # Preprocess test set
    preprocessor = get_preprocessor()

    # Run the test
    test_paraphrase_attacks(args.file_path, model, preprocessor)
